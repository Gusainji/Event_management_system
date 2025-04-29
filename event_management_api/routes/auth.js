const express = require("express");
const router = express.Router();
const { verfiyUserToken } = require('../middlewares/auth.middleware.js');
const {
    login,
    register,
} = require("../controllers/auth.js");

router.post("/login", login);
router.post("/register", register);

module.exports = router;