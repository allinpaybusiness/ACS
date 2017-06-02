# -*- coding: utf-8 -*-
"""
Created on Fri May 26 19:29:26 2017

@author: s
"""

#!coding=UTF-8    
from http.server import HTTPServer,BaseHTTPRequestHandler     
import io,shutil,urllib 
import sys
sys.path.append('D:/万鉴通')  
from fyzpred01 import fyz_pred_00
from fyzpred01 import fyz_pred_01
    
class MyHttpHandler(BaseHTTPRequestHandler):     
    def do_GET(self):     
       # name="World"  
       
        r_str1 = " format error, no ? "
        if '?' in self.path:#如果带有参数     
            self.queryString=urllib.parse.unquote(self.path.split('?',1)[1])     
            #name=str(bytes(params['name'][0],'GBK'),'utf-8')     
            params=urllib.parse.parse_qs(self.queryString)     
            print(params)  
            r_str1 = fyz_pred_01.fyz_pred(params)
            print("allinpay_score == %s " %str(r_str1))
            print("rsk_score == %s " %params['rsk_score'])

            #name=params["name"][0] if "name" in params else None     
      #  r_str="Hello world"
        r_str = str(r_str1)

        enc="UTF-8"    
        encoded = ''.join(r_str).encode(enc)     
        f = io.BytesIO()     
        f.write(encoded)     
        f.seek(0)     
        self.send_response(200)     
        self.send_header("Content-type", "text/html; charset=%s" % enc)     
        self.send_header("Content-Length", str(len(encoded)))     
        self.end_headers()     
        shutil.copyfileobj(f,self.wfile)      
    def do_POST(self):     
        s=str(self.rfile.readline(),'UTF-8')#先解码     
        print(urllib.parse.parse_qs(urllib.parse.unquote(s)))#解释参数     
        self.send_response(301)#URL跳转     
        self.send_header("Location", "/?"+s)     
        self.end_headers()     
    
httpd=HTTPServer(('',8080),MyHttpHandler)     
print("Server started on 127.0.0.1,port 8080.....")     
httpd.serve_forever()   