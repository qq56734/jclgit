import untangle

class LcmReader:

    def __init__(self, lcmpath):
        xml = untangle.parse(lcmpath)
        self.oem_code = xml.Material['OEMCode']
        self.channel_port = xml.Material['ChannelPort']

    @classmethod
    def empty(self):
        self.oem_code = None
        self.channel_port = None
        return self