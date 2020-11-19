import sqlalchemy
from sqlalchemy.orm import sessionmaker
from util import consts
import os
import zipfile
from util.LcmReader import *
from util.models import *

TechnicalFolder = '/Users/yongwei/SynologyDrive/柏楚控软/数据/工艺文件/'

engine_ms = sqlalchemy.create_engine(consts.MSSQL_CONNECTION_MATERIAL)
DBSessionMS = sessionmaker(bind=engine_ms)
session_ms = DBSessionMS()

engine_my = sqlalchemy.create_engine(consts.MYSQL_CONNECTION_FMESQA)
DBSessionMY = sessionmaker(bind=engine_my)
session_my = DBSessionMY()

def main():
    for i in range(100):
        start_id = query_start_id()
        print('Start sync from %d'%start_id)
        materials = query_material_logs(start_id)
        for m in materials:
            save_material(m)
        session_my.commit()

def query_material_logs(start_id):
    query = (session_ms
             .query(CmlLogMaterial)
             .filter(CmlLogMaterial.ID > start_id)
             .limit(100)
             )
    materials = query.all()
    return materials

def query_start_id():
    query = (session_my
             .query(TblTechnicalMaterial)
             .order_by(TblTechnicalMaterial.org_id.desc())
             )
    m_last = query.first()
    if m_last is None:
        return 0
    else:
        return m_last.org_id

def save_material(material):
    m = material
    m_name = m.MaterialName
    m_machine_id = m.MachineID
    m_lcm = LcmReader.empty()
    # 处理文件保存
    if len(m_name) > 0:
        m_data = m.FileData
        technical_folder = os.path.join(TechnicalFolder, str(m_machine_id))
        if not os.path.exists(technical_folder):
            os.makedirs(technical_folder)
        technical_file = os.path.join(technical_folder, m_name)
        f = open(technical_file, 'wb')
        f.write(m_data)
        f.close()
        print('ID:%d create Material File:%s' % (m.ID, technical_file))
        # 处理文件解压读取
        if technical_file.endswith('.fsm'):
            extract_folder = un_zip(technical_file)
            # 读取解压后的 lcm 文件
            lcmpath = os.path.join(extract_folder, "material.lcm")
            if os.path.exists(lcmpath):
                m_lcm = LcmReader(lcmpath)
    else:
        print('ID:%d not exist Material' % m.ID)
    # 处理数据保存
    tm = TblTechnicalMaterial(org_id=m.ID,
                              machine_id=m.MachineID,
                              oem_code=m_lcm.oem_code,
                              channel_port=m_lcm.channel_port,
                              instance_id=m.InstanceID,
                              user_file_name=m.MaterialName,
                              user_file_path=m.FileName,
                              local_file_name=m.MaterialName,
                              local_file_path=str(m.MachineID) + '/' + m_name,
                              user_note=m.UserNote,
                              app_name=m.AppName,
                              action_type=m.Action,
                              action_time=m.ActionTime,
                              )
    session_my.add(tm)


def un_zip(file_name):
    extract_folder = file_name + "_files/"
    """unzip zip file"""
    zip_file = zipfile.ZipFile(file_name)
    zip_file.setpassword(b'www.fscut.com\material')
    if os.path.isdir(file_name + "_files"):
        pass
    else:
        os.mkdir(file_name + "_files")
    for names in zip_file.namelist():
        zip_file.extract(names, extract_folder)
    zip_file.close()
    return extract_folder

if __name__ == '__main__':
    main()