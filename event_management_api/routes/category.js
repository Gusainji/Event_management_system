const express = require("express");
const router = express.Router();
const { verfiyUserToken } = require('../middlewares/auth.middleware.js');
const {
    allCategories
} = require("../controllers/categories.js");

router.get("/all", allCategories);

module.exports = router;