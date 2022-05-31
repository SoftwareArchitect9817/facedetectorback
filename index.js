const express=require('express');
const app=express();
const cors = require('cors');
const bodyParser = require('body-parser');
const Router = require("./router")
const hostname = '127.0.0.1';
const port = 3998;
app.use(cors());
app.use(express.static('./out'));
app.use(bodyParser.json({limit: "15360mb", type:'application/json'}));
app.use(bodyParser.urlencoded({ limit: "15360mb",extended: true }));
app.use('/api', Router);

app.listen(port, () => {
  console.log(`Server running at http://${hostname}:${port}/`);
});
