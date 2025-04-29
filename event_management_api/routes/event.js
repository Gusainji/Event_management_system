const express = require("express");
const router = express.Router();
const { verfiyUserToken } = require('../middlewares/auth.middleware.js');
const multer = require('multer');
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'D:/p/event_management_app/src/assets/events/')
    },
    filename: function (req, file, cb) {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9)
        cb(null, uniqueSuffix + '.png')
    },
    
});
const upload = multer({ storage: storage });
const {
    createEvent,
    getAllEvents,
    getMyEvents,
    getEventById,
    enrollEvent,
    getEnrolledEvents,
    isEnrolled,
    unenroll
} = require("../controllers/event.js");

router.get("/all", getAllEvents);
router.get("/id/:id", getEventById);
router.use(verfiyUserToken);
router.get("/myevent", getMyEvents);
router.post("/enroll/:id", enrollEvent);
router.get("/mytickets", getEnrolledEvents);
router.get("/isenrolled/:id", isEnrolled);
router.post("/unenroll/:id", unenroll);
router.post("/create", upload.single('cover'), createEvent);

module.exports = router;