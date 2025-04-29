const db = require('../db/connection.js');
const {DataTypes} = require('sequelize');

const Event = db.sequelize.define('events', {
    id:{
        type: DataTypes.INTEGER(),
        primaryKey: true,
        autoIncrement: true
    },
    name: {
        type: DataTypes.STRING(64),
        allowNull: false
    },
    description:{
        type: DataTypes.TEXT(),
        allowNull:false
    },
    auther:{
        type: DataTypes.STRING(64),
        allowNull:false,
        foreignKey: true
    },
    start_date:{
        type: DataTypes.BIGINT(),
        allowNull:false
    },
    end_date:{
        type: DataTypes.BIGINT(),
        allowNull:false
    },
    duration:{
        type: DataTypes.BIGINT(),
        allowNull:false
    },
    category_id:{
        type: DataTypes.INTEGER(),
        allowNull:false
    },
    is_online:{
        type: DataTypes.BOOLEAN(),
        allowNull:false
    },
    location:{
        type: DataTypes.STRING(64),
        allowNull:true
    },
    is_free:{
        type: DataTypes.BOOLEAN(),
        allowNull:false
    },
    price:{
        type: DataTypes.FLOAT(),
        allowNull:true
    },
    image:{
        type: DataTypes.STRING(64),
        allowNull:false
    }
})

// db.sequelize.sync();
module.exports = Event;