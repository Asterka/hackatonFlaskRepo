const express = require('express');
const app = express();
const port = process.env.PORT || 3004;
app.use('/', express.static(__dirname + 'hackaton-front/dist/'));
app.listen(port);

console.log('Server started at ' + port);
