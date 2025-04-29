const db = require('../db/connection.js');
const {DataTypes} = require('sequelize');

const Ticket = db.sequelize.define('tickets', {
    id:{
        type: DataTypes.INTEGER(),
        primaryKey: true,
        autoIncrement: true
    },
    user: {
        type: DataTypes.STRING(64),
        allowNull: false
    },
    event:{
        type: DataTypes.INTEGER(),
        allowNull:false
    }
})

// db.sequelize.sync();
module.exports = Ticket;