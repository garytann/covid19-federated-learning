import Head from 'next/head'
import Image from 'next/image'
import { Inter } from 'next/font/google'
import styles from '@/styles/Home.module.css'
import { Button , message, Upload , Modal} from 'antd'
import { useState } from 'react'
import { DownloadOutlined, LaptopOutlined, UploadOutlined } from '@ant-design/icons';


const inter = Inter({ subsets: ['latin'] })


const props = {
  name: 'file',
  action: 'http://localhost:8000/client/inference',
  headers: {
    authorization: 'authorization-text',
  },
  async onChange(info) {
    if (info.file.status !== 'uploading') {
      console.log(info.file, info.fileList);
    }
    if (info.file.status === 'done') {
      message.success(`${info.file.name} file uploaded successfully`);
      console.log(info.file.response);
      Modal.info({
        title: 'Results',
        content: info.file.response
      });
      // try {
      //   const res = await fetch('http://localhost:8000/client/inference',
      //   {
      //     method: 'POST',
      //     headers: {
      //       'Content-Type': 'multipart/form-data'
      //     },
      //   });
        // const data = await res.json();
        // setResponseData(data);
        // console.log(typeof(data));
         // Show the response data in a modal pop-up
        // Modal.info({
        //   title: 'Results',
        //   // content: JSON.stringify(data, null, 2),
        //   content: JSON.parse(data),
        // });
        // Handle the JSON response here as needed
      // } catch (error) {
      //   message.error('Error fetching data:', error);
      //   // Handle the error if necessary
      // }
    } else if (info.file.status === 'error') {
      message.error(`${info.file.name} file upload failed.`);
    }
  },
};

export default function Home() {

  // client training
  const handleClienTraining = async () => {
    try {
      const res = await fetch('http://localhost:8000/client/train');
      const data = await res.json();
      setResponseData(data); // Store the fetched data in state
    } catch (error) {
      console.error('Error training local data:', error);
    }
  };

  // client update
  const handleClienUpdate = async () => {
    try{
      const res = await fetch('http://localhost:8000/client/send_weights',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
        },
        });
      const data = await res.json();
      setResponseData(data); // Store the fetched data in state
    } catch (error){
      console.error('Error updating weight:', error);
    }
  };

  // handle inference
  const handleInference = async () => {
    try{
      const res = await fetch('http://localhost:8000/client/inference',
      
      {
        method: 'POST',
        headers: {
          'Content-Type':'Multipart/form-data'
        },
      });
      const data = await res.json();
      setResponseData(data); // Store the fetched data in state
    }catch (error){
      console.error('Error inference:', error);
    }
  };

  const [size, setSize] = useState('large'); // default is 'middle'
  const [fileList, setFileList] = useState([]);
  const [uploading, setUploading] = useState(false);

  return (
    <>
      <Head>
        <title>COVIDLUS FL</title>
        <meta name="description" content="Federated Learning with CovidLUS" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className={`${styles.main} ${inter.className}`}>
      <h1 className={styles.header}>Federated Learning COVID19 Detection</h1>
        <div className={styles.buttonContainer}>
          <p className={styles.textAboveButton}>Begin local training</p>
          <Button onClick={handleClienTraining} type="primary" className={styles.trainButton}>
            <LaptopOutlined style={{ fontSize: '24px' }} />
            Train
          </Button>
        </div>

        <div className={styles.buttonContainer}>
          <p className={styles.textAboveButton}>Update client weight</p>
          <Button onClick={handleClienUpdate} type="primary" className={styles.trainButton}>
            <DownloadOutlined style={{ fontSize: '24px' }} />
            Update
          </Button>
        </div>

        <div className={styles.buttonContainer}>
          <p className={styles.textAboveButton}>Inference</p>
          <Upload {...props}>
            <Button onClick = {handleInference} className={styles.trainButton} icon={<LaptopOutlined />}>Upload Image</Button>
          </Upload>
          {/* <Button type="primary" className={styles.trainButton}>
            <LaptopOutlined style={{ fontSize: '24px' }} />
            Predict
          </Button> */}
        </div>

      </main>
    </>
  )
}
