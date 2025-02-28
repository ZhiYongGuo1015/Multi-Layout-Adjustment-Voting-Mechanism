import os.path
import paddle
from paddleocr import PaddleOCR
import numpy as np
import cv2 as cv
import xlwt
import xlrd
import re

IMG_PATH = r'Data/3.jpg'

paddle.set_device('gpu')
ocr = PaddleOCR(use_angle_cls=True, use_gpu=True)  # 使用CPU预加载，不用GPU
# ocr = hub.Module(name='chinese_ocr_db_crnn_server', use_gpu=True)


class ImgProcess:
    def __init__(self):
        pass

    def read_img(self, img_path, beyond_thre=255):
        img = cv.imread(img_path)   #使用 OpenCV 读取指定路径的图像文件，并返回图像数据。
        # 灰度化与二值化
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  #将读取的图像转换为灰度图，减少色彩信息，便于后续处理
        binary_img = cv.adaptiveThreshold(gray, beyond_thre, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)  #对灰度图像进行自适应二值化，将像素分为黑（0）和白（255）两类

        return binary_img

    def show_img(self, img):
        #显示图像窗口，并等待用户按键关闭。
        cv.namedWindow('img', cv.WINDOW_NORMAL)  #创建一个可调整大小的窗口
        cv.imshow('img', img) #显示图像
        cv.waitKey(0)  #等待用户按键输入，参数为 0 表示无限等待
        cv.destroyAllWindows()  #销毁所有图像窗口

    def get_h_proj(self, bia_img):
        """
        水平投影
        :param img: 二值化图像
        :return: 列表，每一行白色像素个数
        """
        # 图像高与宽
        h, w = bia_img.shape
        # 长度与图像高度一致的数组
        h_script = [0] * h
        #初始化水平投影列表 h_script，每个元素表示图像对应行的黑色像素数量
        # 循环统计每一行黑像素的个数
        #遍历图像的每个像素，统计每一行中黑色像素（值为 0）的数量，结果存储在 h_script 中
        for y in range(h):
            for x in range(w):
                if bia_img[y, x] == 0:
                    h_script[y] += 1
        return h_script

    def get_v_proj(self, bia_img):
        """
        垂直投影
        :param img: 二值化图像
        :return: 列表，每一列黑色像素个数
        """
        #初始化垂直投影列表 v_script，每个元素表示图像对应列的黑色像素数量
        h, w = bia_img.shape
        v_script = [0] * w
        #遍历图像的每个像素，统计每一列中黑色像素的数量，结果存储在 v_script 中。
        for x in range(w):
            for y in range(h):
                if bia_img[y, x] == 0:
                    v_script[x] += 1
        return v_script

    # 分割文字行
    def spilt_position(self, img):  # 二值化图像
        img = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_CONSTANT, None, 255)
        #为图像添加边框，防止分割过程中文字接触边界导致误判
        h, w = img.shape
        start = 0
        crop = []  # 存储分割行的图像的列表
        h_start = []
        h_end = []
        h_script = self.get_h_proj(img)

        # 根据水平投影获取垂直分割位置
        for i in range(len(h_script)):
            if h_script[i] > 0 and start == 0:
                h_start.append(i)
                start = 1
            if h_script[i] <= 0 and start == 1:
                h_end.append(i)
                start = 0

        # 分割行
        #根据记录的起始和结束位置，将每段黑色像素行从图像中切分出来
        for i in range(len(h_start)):
            crop_img = img[h_start[i]:h_end[i], 0:w]
            crop_img = cv.copyMakeBorder(crop_img, 0, 0, 0, 0, cv.BORDER_CONSTANT, None, 255)
            crop.append(crop_img)

        return crop  #返回分割后的图像行列表


class OcrImg:
    def __init__(self):
        pass

    def get_mode(self, frenquency):  # 获得列表元素及出现频次，生成字典
        dict_fre = {}
        fre_str = []
        for i in frenquency:
            i = ''.join(i)
            fre_str.append(i)
        for i in fre_str:
            if i not in dict_fre.keys():
                dict_fre[i] = fre_str.count(i)

        return dict_fre
        #统计 frenquency 中各个元素（字符串）的出现频次。

    #pand-ocr
    #调整图像边框并识别/过调整图像边框的大小，尝试不同设置下的OCR识别，并返回所有结果。
    # def change_border(self, img, constant, mark):  # mark = 0时输出变白框结果
    #     import pytesseract
    #
    #     # ocr = PaddleOCR(use_angle_cls=True, use_gpu=True)  # 使用CPU预加载，不用GPU
    #
    #     result_dif = []  # 不同白色边框大小识别结果
    #
    #     if mark:
    #         result_1 = []
    #         img = cv.copyMakeBorder(img, constant, constant, constant, constant, cv.BORDER_CONSTANT, None, 255)
    #         text_constant = ocr.ocr(img, cls=True)
    #         # text_constant = ocr.recognize_text(img)
    #         if text_constant != [[]]:
    #             result_1.append(text_constant[0][0][1][0])
    #             result_dif.append(result_1)
    #         #调整图像边框（四周增加宽度为 constant 的白边）。
    #         #使用 OCR (ocr.ocr) 对图像进行文本识别。
    #         #如果识别成功，将结果存入 result_dif。
    #
    #         # text = pytesseract.image_to_string(img, lang='chi_sim')
    #         # print(text)
    #         # result_1.append(text)
    #         # result_dif.append(result_1)
    #
    #
    #     #循环调整边框大小，每次增加 constant = constant + 30。
    #     #使用 OCR 进行识别。
    #     #如果识别成功，将结果存入 result_dif。
    #     #如果识别失败，继续调整边框。
    #     if not mark:
    #         for i in range(9):
    #             result_1 = []
    #             img = cv.copyMakeBorder(img, constant, constant, constant, constant, cv.BORDER_CONSTANT, None, 255)
    #             text = ocr.ocr(img, cls=True)  # 打开图片文件
    #             # text = ocr.recognize_text(img)
    #             if text != [[]]:
    #                 result_1.append(text[0][0][1][0])
    #                 result_dif.append(result_1)
    #                 constant += 30
    #             else:
    #                 constant += 30
    #                 continue
    #
    #     return result_dif


    #tesser-cor
    def change_border(self, img, constant, mark):
        result_dif = []  # 用来存储识别结果
        import pytesseract
        import cv2 as cv
        # 设置 Tesseract 可执行文件的路径（如果已经添加到环境变量中，可以省略）
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows 系统示例路径

        # 如果 mark 为 True，添加边框并进行识别
        if mark:
            result_1 = []
            img = cv.copyMakeBorder(img, constant, constant, constant, constant, cv.BORDER_CONSTANT, None, 255)
            # 使用 pytesseract 识别文本

            text_constant = pytesseract.image_to_string(img, lang='chi_sim')  # 'chi_sim' 是简体中文的语言代码
            if text_constant.strip():  # 如果识别结果非空
                result_1.append(text_constant)
                result_dif.append(result_1)

        # 如果 mark 为 False，循环增加边框大小并进行识别
        if not mark:
            for i in range(9):
                result_1 = []
                img = cv.copyMakeBorder(img, constant, constant, constant, constant, cv.BORDER_CONSTANT, None, 255)
                # 使用 pytesseract 识别文本


                text = pytesseract.image_to_string(img, lang='chi_sim')
                if text.strip():  # 如果识别结果非空
                    result_1.append(text)
                    result_dif.append(result_1)
                    constant += 30  # 如果识别成功，增加 constant 来进一步调整图像边框
                else:
                    constant += 30  # 如果识别失败，继续增加边框大小

        return result_dif

    #从 result_dif 中选择出现频次最高的结果，作为最终的结果。
    def pre_rectify(self, result_dif):  # 获得预纠正的结果
        result_fre_dic = self.get_mode(result_dif)
        base_fre = 0  # 频次最高元素的频次
        base_result = []  # 频次最高的元素
        for i in result_fre_dic.keys():
            if result_fre_dic[i] >= base_fre:
                base_fre = result_fre_dic[i]
                base_result = i
        # print(base_result)
        return base_result

    #图像OCR识别主流程。
    def get_text(self, img, constant, mark):
        if not mark:
            result_dif = self.change_border(img, constant, mark)
            result = self.pre_rectify(result_dif)
            return result
        if mark:
            result = self.change_border(img, constant, mark)
            result = self.pre_rectify(result)
            return result


class GetText:
    def __init__(self):
        self.oi = OcrImg()
        self.ip = ImgProcess()

    def cell_text(self, cell, img, constant, mark, txt_path='Data/text.txt', tolerance=10):
        text_result = []

        cell_cnt = len(cell)
        for i in range(cell_cnt):
            cell_text = []
            x1, x2 = cell[i][0][0], cell[i][1][0]
            y1, y2 = cell[i][0][1], cell[i][2][1]
            cell_img = img[y1 + tolerance: y2, x1 + 12: x2]

            kernel1 = np.ones((2, 2), np.uint8)
            kernel2 = np.ones((2, 2), np.uint8)
            # cell_img = cv.erode(cell_img, kernel1, iterations=1)
            # cell_img = cv.dilate(cell_img, kernel2, iterations=1)
            # self.ip.show_img(cell_img)
            crop = self.ip.spilt_position(cell_img)
            for row in crop:
                row_text = self.oi.get_text(row, constant, mark)
                cell_text.append(row_text)
            text_result.append(cell_text)

        list_txt(txt_path, text_result)
        return text_result

    def associate_text(self, cell, text_result):
        cellNtext = []
        cell_cnt = len(cell)
        for i in range(cell_cnt):
            cell_text = [cell[i], text_result[i]]
            cellNtext.append(cell_text)

        return cellNtext

    def vertical_trophic(self, cell_mark, col_x):
        all_col_group = []  # 按列存储每个点的坐标和文本信息
        text_list = []  # 按列存储文本信息
        col_num = len(col_x) - 1
        for i in range(col_num):
            col_x1, col_x2 = col_x[i], col_x[i + 1]  # 一列的左右x坐标
            col_group = []  # 存储该列所有的单元格（包括包含该列的单元格）
            for j in cell_mark:
                x1, x2 = j[0][0][0], j[0][1][0]
                if x1 <= col_x1 and x2 >= col_x2:
                    col_group.append(j)
            col_group = sorted(col_group, key=lambda x: x[0][0][1], reverse=False)
            all_col_group.append(col_group)

        return all_col_group

    def horizontal_trophic(self, cellNtext, col_group, all_col_group):
        top, btm = cellNtext[0][0][1], cellNtext[0][2][1]  # top and btm y coordinate of table head
        include_cnt = 0  # get cell cnt on horizontal direction of this cell
        col_cnt = len(all_col_group)

        for i in range(col_cnt):
            top_mark, btm_mark = 0, 0
            col = all_col_group[i]
            if col == col_group or len(col) == 1:
                pass

            else:
                cnt = 0
                for j in col:
                    text = j[1]
                    y1, y2 = j[0][0][1], j[0][2][1]
                    if text and text != [[]] and text != ' ':
                        if top == y1:  # include leaf cell properly
                            top_mark = 1
                        if btm == y2:
                            btm_mark = 1
                        if top <= y1 < y2 <= btm:
                            cnt += 1
                if cnt > include_cnt and top_mark == btm_mark == 1:
                    include_cnt = cnt

        if include_cnt == 0:
            include_cnt = 1

        return include_cnt

    def head_trophic(self, cell_mark, all_col_group):
        cell_1st = cell_mark[0]
        head_top, head_btm = cell_1st[0][0][1], cell_1st[0][2][1]  # top&btm range of table head
        text_list = []
        col_cnt = len(all_col_group)
        for i in range(col_cnt):
            col_text = []
            col_group = all_col_group[i]

            for j in col_group:
                top, btm = j[0][0][1], j[0][2][1]
                if head_top <= top < btm <= head_btm:
                    include_cnt = self.horizontal_trophic(j, col_group, all_col_group)
                else:
                    include_cnt = 1

                for m in range(include_cnt):
                    col_text.append(j[1])
            text_list.append(col_text)

        return text_list

    # 钻孔柱状图主表信息抽取
    def info2cvs(self, text_list, xls_path='6.xls'):
        book = xlwt.Workbook(encoding='gbk')
        sheet1 = book.add_sheet('borehole')
        col_len = len(text_list)
        for i in range(col_len):
            for j in range(len(text_list[i])):
                for m in range(len(text_list[i][j])):
                    if not text_list[i][j][m]:
                        text_list[i][j][m] = ' '
                text = ''.join(text_list[i][j])
                sheet1.write(j, i, text)

        book.save(xls_path)

    def generate_xls(self, cell, text_list, col_x, xls_path, cellNtext_path):
        cellNtext = self.associate_text(cell, text_list)
        list_txt(cellNtext_path, cellNtext)
        all_col_group = self.vertical_trophic(cellNtext, col_x)
        result_list = self.head_trophic(cellNtext, all_col_group)
        self.info2cvs(result_list, xls_path)


#读取文件并将其内容作为列表返回，或者将列表保存到文件中。
def list_txt(path, list=None):
    # path: 表示文件路径
    '''
    :return:
    :param path: 储存list的位置
    :param list: list数据:
    return: None/relist 当仅有path参数输入时为读取模式将txt读取为list当path参数和list都有输入时为保存模式将list保存为txt
    '''
    if list != None:
        file = open(path, 'w', encoding='utf-8')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r', encoding='utf-8')
        rdlist = eval(file.read())
        file.close()
        return rdlist

def get_word(img, constant, mark, txt_path, tolerance=10):

#def get_word(img, constant, mark, txt_path='Data/text.txt', tolerance=10):
    #从输入图像中提取文字行，并使用 OCR（光学字符识别）提取文字内容，将识别结果保存到指定路径的文本文件中
    text_result = []
    ip = ImgProcess()
    oi = OcrImg()
    crop = ip.spilt_position(img)  #使用 ImgProcess 类的 spilt_position 方法，将图像分割成独立的行
    for row in crop:
        row_text = oi.get_text(row, constant, mark) #遍历分割后的图像行 crop，对每一行图像进行 OCR 识别
        text_result.append(row_text)  #将识别的结果添加到 text_result 列表中

    # list_txt(txt_path, text_result)
    # for row in crop:
    #     #再次对 crop 列表中的每一行执行 OCR 识别，逻辑重复
    #     row_text = oi.get_text(row, constant, mark)
    #     text_result.append(row_text)
        # ip.show_img(row)
    f = open(txt_path, 'w')
    #以写入模式打开指定的文本文件 txt_path
    for i in text_result:
        if i:
            f.write(i)
            # f.write('\n') #可在每行结果后添加换行符。
    return text_result
# 从图像中提取单元格（cell）区域，并将其保存为单独的图像文件


def get_cell_image(cell, img_path, tolerance=20):
    # cell = list_txt('Data/cell.txt')

    #读取图像并转换为灰度图像
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 全局二值化
    thres, img = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
    #对灰度图像进行全局二值化

    #遍历 cell 列表，提取每个单元格的图像区
    for i in range(len(cell)):

        x1, x2 = cell[i][0][0] + tolerance, cell[i][1][0]
        y1, y2 = cell[i][0][1] + tolerance, cell[i][2][1] #获取单元格的坐标并裁剪图像
        cell_img = img[y1: y2, x1: x2]

        #加上 tolerance 扩展或收缩边界
        kernel1 = np.ones((2, 2), np.uint8)
        kernel2 = np.ones((2, 2), np.uint8)
        #腐蚀：减少图像中的白色区域。 膨胀：扩展图像中的白色区域
        cell_img = cv.erode(cell_img, kernel1, iterations=1)
        cell_img = cv.dilate(cell_img, kernel2, iterations=1)

        #path1 = 'cell_img/'   #保存图像的目录
        path1 = 'test - 202412/'   #保存图像的目录
        path2 = str(i) + '.jpg'
        save_path = os.path.join(path1, path2)
        cv.imwrite(save_path, cell_img) #使用 cv.imwrite 将裁剪的单元格图像 cell_img 保存到 save_path


def get_row_img(img_path='E:\Boreholelog_Constructrize\shp_borehole\Image_extraction\cell_img/51.jpg'):
    img_process = ImgProcess()
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    crop = img_process.spilt_position(img)
    row_cnt = len(crop)
    for i in range(row_cnt):
        path1 = 'row_position/'
        path2 = str(i) + '.jpg'
        save_path = os.path.join(path1, path2)
        cv.imwrite(save_path, crop[i])


# def word_compare(img_dic='Acc_result/rock_col2/', save_dic='Acc_result/T02_TEXT_TES2/'):
#     imgs_name = os.listdir(img_dic) #获取输入目录中的图像文件名

def word_compare(img_dic='Acc_result/rock_col3/', save_dic='Acc_result/T02_TEXT_TES3/'):
    imgs_name = os.listdir(img_dic)  # 获取输入目录中的图像文件名
    for i in imgs_name:
        print(i)   # 获取 img_dic 中的所有文件名，并打印每个文件名。
        img_path = img_dic + i  #i: 当前处理的文件名
        img_id = i.split('.')[0]  #提取图像文件名的主文件名部分（去掉扩展名）

        # preprocess
        #对图像进行预处理，包括灰度化和二值化。
        ip = ImgProcess()
        img = cv.imread(img_path)  #当前图像的完整路径
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        retval, img = cv.threshold(img, 200, 255, cv.THRESH_BINARY)  # scan digtal image
        # ip.show_img(img)

        path_0 = save_dic + img_id + '_0.txt'
        #定义保存路径
        path_30 = save_dic + img_id + '_30.txt'
        path_60 = save_dic + img_id + '.60.txt'
        path_90 = save_dic + img_id + '.90.txt'
        path_120 = save_dic + img_id + '.120.txt'
        path_150 = save_dic + img_id + '.150.txt'
        path_text = save_dic + img_id + '_text.txt'

        get_word(img, 0, 1, path_0)  #1: 标志位，指定 get_word 的工作模式（如是否使用边框调整）。
        get_word(img, 30, 1, path_30)
        # get_word(img, 60, 1, path_60)
        # get_word(img, 90, 1, path_90)
        # get_word(img, 120, 1, path_120)
        # get_word(img, 150, 1, path_150)
        # get_word(img, 30, 0, path_text)

#处理指定目录中的文本文件，逐行读取内容，并将每行内容保存到一个新目录下的文件中。如果某一行为空，则用“空null！！！”代替。

def handle_txt(txt_dic='Acc_result/rock_text/text_30/text/', result_dic='Acc_result/rock_text/text_30/text_line/'):
    txts_name = os.listdir(txt_dic)
    #获取 txt_dic 目录中所有文件的文件名，并存储到 txts_name 列表中
    #遍历每个文件并读取内容
    for i in txts_name:
        txt_path = txt_dic + i
        list = list_txt(txt_path)

        save_path = result_dic + i
        with open(save_path, 'w') as file:
            # 遍历列表中的每个元素并逐行写入文件
            for j in list:
                if j:
                    file.write(j)
                else:
                    file.write('空null！！！')
                file.write('\n')

#def get_result_T02(constant, mark, txt_dic='C:/Users/Weibao/Desktop/THESIS_doc/Result_correct/Data/rock_col2/', result_path='C:/Users/Weibao/Desktop/THESIS_doc/Result_correct/T02_text/'):
def get_result_T02(constant, mark, txt_dic='C:/Users/11921/PycharmProjects/ocr_picture/Image_extraction/Acc_result/rock_col3/',result_path='C:/Users/11921/PycharmProjects/ocr_picture/Image_extraction/Acc_result/T02_text/'):

    imgs_name = os.listdir(txt_dic)
    txt_path = result_path + 'T02_change.txt'
    f = open(txt_path, 'w')
    for img_name in imgs_name:
        print(img_name)
        img_id = img_name.split('.')[0]
        print(img_id, type(img_id))
        img_path = txt_dic + img_name
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        retval, img = cv.threshold(img, 200, 255, cv.THRESH_BINARY)  # scan digtal image
        text_result = []
        ip = ImgProcess()
        oi = OcrImg()
        crop = ip.spilt_position(img)
        for row in crop:
            row_text = oi.get_text(row, constant, mark)
            text_result.append(row_text)
            # ip.show_img(row)

        f.write(f"Image ID: {img_id}\n")  # 写入图像ID，换行
        if not text_result:
            f.write('No text detected\n')  # 如果没有检测到文本，写入提示信息
        else:
            for i in text_result:
                # 检查i是否为列表类型，如果是，转换为字符串
                if isinstance(i, list):
                    for sub_item in i:
                        f.write(str(sub_item))  # 如果是列表，逐个写入
                        f.write('\n')
                else:
                    f.write(i)  # 如果i是字符串，直接写入
                    f.write('\n')  # 每个文本后加换行符
        # f.write(img_id)
        # for i in text_result:
        #     if i:
        #         f.write(i)
        #         f.write('\n')


if __name__ == '__main__':
    # ip = ImgProcess()
    # img = cv.imread('Data/T01_0028_2.jpg')
    # img = cv.imread('Data/T02_43_1.jpg')
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  #将读取的彩色图像转换为灰度图像，为后续的二值化处理做准备
    # retval, img = cv.threshold(img, 200, 255, cv.THRESH_BINARY)  # scan digtal image # 对灰度图像进行二值化处理，将像素值大于 200 的部分设为 255（白色），小于等于 200 的部分设为 0（黑色）。用于生成清晰的二值图像
    # cv.imwrite('temp/image.jpg', img)     #将处理后的二值化图像保存到 temp/image.jpg 文件中，以供后续使用
    # ip.show_img(img)   #调用 ImgProcess 类的 show_img 方法，在一个窗口中显示当前处理后的图像。
    #
    # get_word(img, 30, 0)   #提取图像中的文字行。constant=30 指边框扩展大小，mark=0 表示仅扩展边框，不进行其他操作  使用 OCR 提取文字行，将结果保存到默认路径 Data/text.txt


    word_compare()    #比较不同图像样本在各种边框大小下的 OCR 识别结果，并将这些结果保存到指定路径中
    # handle_txt()      #处理文本文件中的内容，将 OCR 的多次识别结果整理成可读性更高的格式并保存。
    # get_cell_image(IMG_PATH)   #从主图像中提取表格的单元格图像，并将其保存到 cell_img/ 目录下
    # word_compare()    #比较不同图像样本在各种边框大小下的 OCR 识别结果，并将这些结果保存到指定路径中
    #
    # get_result_T02(30, 0)   # 针对数据集 rock_col2 执行 OCR 操作，将结果保存到指定目录路径中
    #
    # cell = list_txt('tempdir/T01/T01_0040/cell.txt')   #读取单元格位置信息（以列表形式存储）并加载到变量 cell 中
    # body_path = 'tempdir/T01/T01_0040/body.jpg'
    # get_cell_image(cell, body_path)   #根据读取的单元格坐标，裁剪出对应单元格的图像内容并保存到指定路径。




