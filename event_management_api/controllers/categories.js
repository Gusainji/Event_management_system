const db = require("../db/connection.js");
const Category = require("../models/categories.js");

async function allCategories(req, res) {
    try{
        const categories = await Category.findAll();
        res.status(200).json(categories);
    }
    catch(e){

    }
}

module.exports = {
    allCategories
}