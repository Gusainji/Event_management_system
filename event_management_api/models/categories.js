const db = require('../db/connection.js');
const {DataTypes} = require('sequelize');

const Category = db.sequelize.define('categories', {
    id:{
        type: DataTypes.INTEGER(),
        primaryKey: true
    },
    name: {
        type: DataTypes.STRING(64),
        allowNull: false
    }
})

// db.sequelize.sync();
module.exports = Category;