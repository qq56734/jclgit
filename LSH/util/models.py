# coding: utf-8
from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.schema import FetchedValue
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()

class CmlLogMaterial(db.Model):
    __tablename__ = 'cml_log_material'

    ID = db.Column(db.Integer, primary_key=True)
    InstanceID = db.Column(db.Integer, nullable=False)
    Action = db.Column(db.Integer, nullable=False)
    ActionTime = db.Column(db.DateTime)
    MaterialName = db.Column(db.Unicode(1000))
    FileName = db.Column(db.Unicode(1000), nullable=False)
    FileData = db.Column(db.LargeBinary)
    SyncTime = db.Column(db.DateTime)
    PierceStepCount = db.Column(db.Integer, server_default=db.FetchedValue())
    PierceStyle = db.Column(db.Integer, server_default=db.FetchedValue())
    UseStepPunch = db.Column(db.INT)
    Thickness = db.Column(db.Float(53))
    UserNote = db.Column(db.Unicode(2048))
    UsePierce = db.Column(db.INT)
    UsePrePierce = db.Column(db.INT)
    UseCoverCut = db.Column(db.INT)
    UsePathCool = db.Column(db.INT)
    WorkSpeed = db.Column(db.Float(53))
    ChannelPort = db.Column(db.Integer)
    UseExtFollow = db.Column(db.INT)
    UseStaticFollow = db.Column(db.INT)
    UseSlowLead = db.Column(db.INT)
    UseSlowEnd = db.Column(db.INT)
    UseDynamicFreq = db.Column(db.INT)
    UseDynamicPwm = db.Column(db.INT)
    MachineID = db.Column(db.Integer)
    AppName = db.Column(db.String(100, 'Chinese_PRC_CI_AS'))



class TblTechnicalMaterial(db.Model):
    __tablename__ = 'tbl_technical_material'

    id = db.Column(db.Integer, primary_key=True)
    org_id = db.Column(db.Integer, nullable=False)
    instance_id = db.Column(db.Integer, nullable=False)
    machine_id = db.Column(db.Integer, nullable=False)
    oem_code = db.Column(db.String(255, 'utf8mb4_general_ci'))
    channel_port = db.Column(db.Integer)
    user_file_name = db.Column(db.String(255, 'utf8mb4_general_ci'))
    user_file_path = db.Column(db.String(255, 'utf8mb4_general_ci'))
    local_file_name = db.Column(db.String(255, 'utf8mb4_general_ci'))
    local_file_path = db.Column(db.String(255, 'utf8mb4_general_ci'))
    user_note = db.Column(db.String(255, 'utf8mb4_general_ci'))
    app_name = db.Column(db.String(255, 'utf8mb4_general_ci'))
    action_type = db.Column(db.Integer)
    action_time = db.Column(db.DateTime)
    create_time = db.Column(db.DateTime, server_default=db.FetchedValue())
    update_time = db.Column(db.DateTime)
    create_by = db.Column(db.String(255, 'utf8mb4_general_ci'))
    update_by = db.Column(db.String(255, 'utf8mb4_general_ci'))
    is_delete = db.Column(db.Integer, server_default=db.FetchedValue())
