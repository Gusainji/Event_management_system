exports.isFileValid = (req, res, next) => {
    const array_of_allowed_file_types = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
    if (!array_of_allowed_file_types.includes(req.memetype)) {
      throw Error('Invalid file');
    }
}
