const Sequelize = require("sequelize");
const DB_CONFIG = require("./configuration.js");

const sequelize = new Sequelize(
  DB_CONFIG.DB,
  DB_CONFIG.USER,
  DB_CONFIG.PASSWORD,
  {
    host: DB_CONFIG.HOST,
    dialect: DB_CONFIG.dialect,
    define: {
      timestamps: false
    }
  }
);

sequelize
  .authenticate()
  .then(() => {
    console.log("Connection has been established successfully.");
  })
  .catch(error => {
    console.error("Unable to connect to the database: ", error);
  });

const db = {};

db.sequelize = sequelize;
db.Sequelize = Sequelize;

module.exports = db;