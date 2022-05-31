const express = require('express');
const router = express.Router();
const tscontrol = require("./controller/tscontrol");
const FileUploader = require("./middleware/Uploader")
const Uploader = new FileUploader(__dirname + "/input")
const multer = require("multer")

router.post("/upload", multer({ storage: Uploader.storage, fileFilter: Uploader.filter }).any(), tscontrol.upload)

module.exports = router;