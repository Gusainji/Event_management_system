const configs = require("../configs");
var jwt = require('jsonwebtoken');

exports.verfiyUserToken = (req, res, next) => {
    let token = req.headers.authorization;
    if (!token) return res.status(401).json({ message: "Access Denied / Unauthorized request" });
    try{
        token = token.split(' ')[1]
        if (token == 'null' || !token) return res.status(401).send({ message: 'Unauthorized request'});
        let verifiedUser = jwt.verify(token, configs.SECRET_KEY);
        if (!verifiedUser) return res.status(401).send({ message: 'Unauthorized request'});
        next();
    }
    catch{
        res.status(400).json({
            message:'invalid token'
        })
    }
}