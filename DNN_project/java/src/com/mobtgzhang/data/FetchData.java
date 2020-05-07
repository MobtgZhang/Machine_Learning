package com.mobtgzhang.data;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLConnection;
import java.util.logging.Logger;

public class FetchData {
    private static Logger logger = Logger.getLogger(String.valueOf(FetchData.class));
    public static File downloadFile(String urlPath, String fileSavePath) throws Exception {

        logger.info("进入下载文件工具类");
        File file = null;
        BufferedInputStream bin = null;
        OutputStream out = null;
        try {
            // 统一资源
            URL url = new URL(urlPath);
            // 连接类的父类，抽象类
            URLConnection urlConnection = url.openConnection();
            // http的连接类
            HttpURLConnection httpURLConnection = (HttpURLConnection) urlConnection;
            // 设定请求的方法，默认是GET
            httpURLConnection.setRequestMethod("GET");
            // 设置字符编码
            httpURLConnection.setRequestProperty("Charset", "UTF-8");
            // 打开到此 URL 引用的资源的通信链接（如果尚未建立这样的连接）。
            httpURLConnection.connect();

            // 文件大小
            int fileLength = httpURLConnection.getContentLength();

            // 文件名
            String filePathUrl = httpURLConnection.getURL().getFile();
            String fileFullName = filePathUrl.substring(filePathUrl.lastIndexOf(File.separatorChar) + 1);
            fileFullName = fileFullName.substring(fileFullName.lastIndexOf("/") + 1);
            logger.info("开始下载文件：" + fileFullName);
            logger.info("file length---->" + fileLength);

            url.openConnection();

            bin = new BufferedInputStream(httpURLConnection.getInputStream());

            String path = fileSavePath;
            file = new File(path);
            if (!file.getParentFile().exists()) {
                file.getParentFile().mkdirs();
            }
            out = new FileOutputStream(file);
            int size = 0;
            int len = 0;
            byte[] buf = new byte[1024];
            while ((size = bin.read(buf)) != -1) {
                len += size;
                out.write(buf, 0, size);
                // 打印下载百分比
                if ((len * 100 / fileLength)>50&&(len * 100 / fileLength)<55) {
                    logger.info("下载了-------> " + len * 100 / fileLength + "%");
                }else    if ((len * 100 / fileLength)>=98) {
                    logger.info("下载了-------> " + len * 100 / fileLength + "%");
                }
                //logger.info("下载了-------> " + len * 100 / fileLength + "%");
            }
            return file;
        } catch (Exception e) {
            // TODO Auto-generated catch block
            throw new Exception(e.toString()+"文件下载异常，请检查下载链接$$"+urlPath);
        } finally {
            if (bin!=null) {
                bin.close();
            }
            if (out!=null) {
                out.close();
            }
        }
    }
}
