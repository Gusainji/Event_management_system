const express = require("express");
const cors = require("cors");
const app = express();

const authRoute = require('./routes/auth');
const categoryRoute = require('./routes/category');
const eventRoute = require('./routes/event');

app.use(cors());
app.use(express.urlencoded({ extended: false }));
app.use(express.json());

app.use("/api/auth", authRoute);
app.use("/api/categories", categoryRoute);
app.use("/api/events", eventRoute);

app.listen(3000, ()=>{
    "Server running"
});