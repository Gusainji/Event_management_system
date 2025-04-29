const db = require("../db/connection.js");
const User = require('../models/user.js');
const bcrypt = require('bcrypt');
var jwt = require('jsonwebtoken');

const { SECRET_KEY } = require('../configs.js');

async function login(req, res) {
    try {
        const { username, password } = req.body;
        const user = await User.findByPk(username);
        const passwordMatch = await bcrypt.compare(password, user.password);
        var token = jwt.sign({
            'username': user.username,
            'name': user.name
        },
            SECRET_KEY,
            {
                algorithm: 'HS512',
                expiresIn: '30d'
            }
        );
        if (passwordMatch) {
            res.status(200).json({
                'message': 'Login successfull',
                'token': token
            })
        }
        else {
            res.status(401).json({
                'message': 'Login failed'
            })
        }
    }
    catch (error) {
        res.status(500).json(error);
    }
}

async function register(req, res) {
    try {
        const {
            username,
            name,
            password
        } = req.body;
        const hash = await bcrypt.hash(password, 10);
        const newUser = User.build({
            'username': username,
            'name': name,
            'password': hash
        });
        await newUser.save();
        res.status(201).json({ success: true });
    }
    catch (error) {
        res.status(500).json(error);
    }
}

module.exports = {
    login,
    register
}