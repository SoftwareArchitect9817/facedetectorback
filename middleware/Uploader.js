const multer = require('multer')

class Uploader {
  constructor(filePath) {
    this.assetsPath = filePath
    this.storage = multer.diskStorage({
      destination: function(req, file, cb) {
        cb(null, filePath)
      },
      filename: function(req, file, cb) {
        const type = '.' + file.mimetype.split("/")[1]
        cb(null, `${Date.now()}${type}`)
      }
    })
  }
}

/**
 * @member {multer.diskStorage} storage
 */
 module.exports  = Uploader