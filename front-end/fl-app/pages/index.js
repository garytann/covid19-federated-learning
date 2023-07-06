import Head from 'next/head'
import Image from 'next/image'
import { Inter } from 'next/font/google'
import styles from '@/styles/Home.module.css'
import { Button } from 'antd'
import { useState } from 'react'
import { DownloadOutlined } from '@ant-design/icons';


const inter = Inter({ subsets: ['latin'] })

export default function Home() {
  const [size, setSize] = useState('large'); // default is 'middle'
  return (
    <>
      <Head>
        <title>Create Next App</title>
        <meta name="description" content="Generated by create next app" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className={`${styles.main} ${inter.className}`}>
        <div>
          <Button type="primary">Train</Button>
        </div>

        <div>
          <Button type="primary" size={size}>Send</Button>
        </div>

        <div>
          <Button type="primary">Predict</Button>
        </div>
      </main>
    </>
  )
}
