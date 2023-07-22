// import formidable from 'formidable';

// export const config = {
//   api: {
//     bodyParser: false,
//   },
// };

// export default async (req, res) => {
//   const form = new formidable.IncomingForm();
//   form.uploadDir = "./";
//   form.keepExtensions = true;
//   form.parse(req, (err, fields, files) => {
//     console.log(err, fields, files);
//   });
// }
import formidable from 'formidable';
import fs from 'fs';

export const config = {
  api: {
    bodyParser: false, // Disable built-in bodyParser to handle it manually with formidable
  },
};

export default function handler(req, res) {
  if (req.method === 'POST') {
    const form = new formidable.IncomingForm();

    form.parse(req, (err, fields, files) => {
      if (err) {
        console.error('Error parsing form:', err);
        res.status(500).json({ error: 'Error parsing form data' });
        return;
      }

      // Here, you can handle the uploaded file in the "files" object
      // For example, move the file to a desired location on the server
      // fs.renameSync(files.file.path, '/path/to/desired/location/' + files.file.name);

      res.status(200).json({ message: 'File uploaded successfully' });
    });
  } else {
    res.status(405).json({ error: 'Method not allowed' });
  }
}