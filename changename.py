# -*- coding: utf-8 -*-
"""
 @Time    : 19-8-20 下午9:39
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : changename.py
"""
import os

class BatchRename():
    def __init__(self,path,word_id):
        self.path = path
        self.word_id = word_id

    def raname(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 5011
        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), 'fimg_'+str(i) + '.jpg')
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    txtName = 'fimg_'+str(i)+'.txt'
                    img_name = 'fimg_'+str(i)+'.jpg'
                    self.writeTxt(txtName,img_name)
                    i = i + 1
                except:
                    continue
        print ('total %d to rename & converted %d jpgs' % (total_num, i))

    def writeTxt(self, txtName, img_name):
        filename = os.path.join(self.path, txtName)
        file = open(filename, 'w')
        line = img_name + ', ' + str(self.word_id)
        print(line)
        file.write(line)
        file.close( )
        return True


if __name__ == '__main__':
    path = './32_2'
    demo = BatchRename(path=path,word_id = 32)
    demo.raname()

