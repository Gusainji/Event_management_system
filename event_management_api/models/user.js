const db = require('../db/connection.js');
const {DataTypes} = require('sequelize');

const User = db.sequelize.define('users', {
    username:{
        type: DataTypes.STRING(64),
        primaryKey: true
    },
    name: {
        type: DataTypes.STRING(64),
        allowNull: false
    },
    password:{
        type: DataTypes.STRING(64),
        allowNull: false
    }
})

// db.sequelize.sync();
module.exports = User;