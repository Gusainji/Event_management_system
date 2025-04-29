const { where } = require("sequelize");
const db = require("../db/connection.js");
const Event = require("../models/event.js");
const Ticket = require("../models/ticket.js");
var jwt = require('jsonwebtoken');

async function createEvent(req, res) {
    try{
        const { event_name, description, category, start_time, duration,end_date, is_online, location, is_free , price } = req.body;
        
        let token = req.headers.authorization;
        const newEvent = Event.build({
            'id': null,
            'name': event_name,
            'description': description,
            'auther': await getAuther(token),
            'start_date': start_time,
            'end_date': end_date,
            'duration': duration,
            'category_id': category,
            'is_online': is_online,
            'location': location,
            'is_free': is_free,
            'price': price,
            'image': req.file.filename
        });
        console.log(newEvent);
        await newEvent.save();
        res.status(201).json(newEvent);
    }
    catch(e){

    }
}

async function getAllEvents(req, res) {
    try{
        const events = await Event.findAll();
        events.sort((a,b) => {
            return a.start_date - b.start_date;
        });
        res.status(200).json(events);
    }
    catch(e){
    }
}

async function getMyEvents(req, res) {
    try{
        let auther = await getAuther(req.headers.authorization);
        const events = await Event.findAll({
            where: {
                auther: auther
            }
        });
        res.status(200).json(events);
    }
    catch(e){
        console.log(e);
    }
}

async function getEnrolledEvents(req, res) {
    try{
        let user = await getAuther(req.headers.authorization);
        const tickets = await Ticket.findAll({
            where: {
                user: user
            }
        });
        let events = [];
        for(let i = 0; i < tickets.length; i++){
            let event = await Event.findByPk(tickets[i].event);
            events.push(event);
        }
        res.status(200).json(events);
    }
    catch(e){
        console.log(e);
    }
}

async function enrollEvent(req, res) {
    try{
        let auther = await getAuther(req.headers.authorization);
        const event = await Event.findByPk(req.params.id);
        if(event.auther == auther){
            res.status(401).json({message: "You can't enroll in your own event"});
        }
        else{
            const newTicket = Ticket.build({
                'id': null,
                'user': auther,
                'event': req.params.id
            });
            await newTicket.save();
            res.status(200).json({message: "Enrolled in event"});
        }
    }
    catch(e){
        console.log(e);
    }
}

async function unenroll(req,res){
    try{
        let user = await getAuther(req.headers.authorization);
        const ticket = await Ticket.findOne({
            where: {
                user: user,
                event: req.params.id
            }
        });
        await ticket.destroy();
        res.status(200).json({message: "Unenrolled from event"});
    }
    catch(e){
        console.log(e);
    }
}

async function isEnrolled(req, res) {
    try{
        let user = await getAuther(req.headers.authorization);
        const ticket = await Ticket.findOne({
            where: {
                user: user,
                event: req.params.id
            }
        });
        if(ticket){
            res.status(200).json(true);
        }
        else{
            res.status(200).json(false);
        }
    }
    catch(e){
        console.log(e);
    }
}

async function getEventById(req,res){
    try{
        const event = await Event.findByPk(req.params.id);
        res.status(200).json(event);
    }
    catch(e){
        console.log(e);
    }
}

async function getAuther(token){
    let user = jwt.decode(token.split(' ')[1]).username;
    console.log(user);
    return user;
}

module.exports = {
    createEvent,
    getAllEvents,
    getMyEvents,
    getEventById,
    enrollEvent,
    getEnrolledEvents,
    isEnrolled,
    unenroll
}