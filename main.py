#!/usr/bin/python3.6.9
# -*- coding: UTF-8 -*-

"""
    @file   main.py
    
    @brief  
    @date   10.03.23 
    @author Thomas Matre
"""

import can, time, json, threading, os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Bytter working directory til den nåværende slik at programmet kan startes utenfra mappa
from drivers.network_handler import Network 
from drivers.STTS75_driver import STTS75; 
from drivers.camPWM import ServoPWM; 
from drivers.camHandler import gstreamerPipe
from functions.fFormating import getBit, getByte, getNum, setBit, toJson
from functions.fPacketBuild import packetBuild
from functions.fNetcallParsing import int8Parse, int16Parse, int32Parse, int64Parse, uint8Parse, uint16Parse, uint32Parse, uint64Parse, fuselightParse, sensorflagsParse, regParamsParse, regflagsParse
from functions.fCancallParsing import canint16Parse, canint8Parse, canSensorAlarmsParse, canuint16Parse, canuint8Parse, canHBParse, can12VParse
from controller.MPC import MPCController, MPCParameters, mpc_step

#gi.require_version("Gst", "1.0")
#from gi.repository import Gst, GLib

#Packets recived from topside and sent to ROV
JBCSHSAC      = 31
REGCONTROL    = 32
ROVCMD        = 33
MANICMD       = 34
ROVCMDSPARE   = 40
MANICMDSPARE  = 41
REGPARAM      = 42
SENSORFLAGS   = 66
SYS5VFLAGS    = 97 
THR12VFLAGS   = 98
MANI12VFLAGS  = 99
canSendDict  = {
      JBCSHSAC : int8Parse,
      REGCONTROL:   regflagsParse,
      ROVCMD:       int8Parse,
      MANICMD:      int8Parse,
      ROVCMDSPARE:  int8Parse,
      MANICMDSPARE: int8Parse,
      REGPARAM:     regParamsParse,
      SENSORFLAGS:  sensorflagsParse,
      SYS5VFLAGS:   fuselightParse,
      THR12VFLAGS:  fuselightParse,
      MANI12VFLAGS: fuselightParse
    }

#Packets recived from ROV over canbus to be parsed and sent to TOPSIDE
THRUSTPAADRAG = 129
REGTEMP       = 130
AKSELERASJON  = 135
GYRO          = 136
MAGNETOMETER  = 137
VINKLER       = 138
TEMPDYBDE     = 139
SENSORERROR   = 140
DATA12VRIGHT  = 150
DATA12VLEFT   = 151
DATA5V        = 152
HBREG         = 155
HBSENSOR      = 156
HB12VMAN      = 157
HB12VTHR      = 158
HB5V          = 159

canReciveDict = {
    THRUSTPAADRAG:  canint8Parse,
    REGTEMP:        canint16Parse,
    AKSELERASJON:   canint16Parse,
    GYRO:           canint16Parse,
    MAGNETOMETER:   canint16Parse,
    VINKLER:        canint16Parse,
    TEMPDYBDE:      canint16Parse,
    SENSORERROR:    canSensorAlarmsParse,
    DATA12VRIGHT:   can12VParse,
    DATA12VLEFT:    can12VParse,
    DATA5V:         canint16Parse,
    HBREG:          canHBParse,
    HBSENSOR:       canHBParse,
    HB12VMAN:       canHBParse,
    HB12VTHR:       canHBParse,
    HB5V:           canHBParse
  }

#Functions recived from topside
CAMERA = 200
#Actions recived from topside
TILT = 'tilt'
START = 'start'
STOP = 'stop'
#Function dict placed inside nettcallback method

# Reads data from network port
def netThread(netHandler, netCallback, flag):
  print("Server started\n")
  flag['Net'] = True
  while flag['Net']:
    try:
      msg = netHandler.receive()
      if msg == b"" or msg is None:
        continue
      else:
        netCallback(msg)
    except ValueError as e:
      print(f'Feilkode i network thread feilmelding: {e}\n\t{msg}')
      break
  netHandler.exit()
  print(f'Network thread stopped')

#Sends heartbeat and alarm if no response in 1sec.
def hbThread(netHandler, canSend, systemFlag, ucFlags):
  print("Heartbeat thread started")
  hbIds = [63, 95 ,125, 126, 127]
  while systemFlag['Can'] and systemFlag['Net']:
    for flag in ucFlags:
      ucFlags[flag] = False
    for id in hbIds:
      canSend(id)
      time.sleep(0.1)
    time.sleep(1)
    for flag in ucFlags:
      if not ucFlags[flag]:
        msg = toJson({"Alarm": f"uC {flag} not responding on CANBus"})
        netHandler.send(msg)
        time.sleep(0.2)
  print("Heartbeat thread stopped")

#Pulls temp from STTS75 every second.
def i2cThread(netHandler, STTS75, systemFlag):
  print("i2c Thread started")
  while systemFlag['Net']:
    temp = STTS75.read_temp()
    msg = toJson({'145': (temp)})
    netHandler.send(msg)
    time.sleep(2)
  print("i2c Thread stopped")

# Computes desired velocities based on position error
def mpcThread(mpcController, mpcParameters, systemFlag, sendCanPacket):
  print("MPC Thread started")
  while systemFlag['Can']:
    if mpcParameters.updateParams:
        mpcController = MPCController(mpcParameters)
        mpcParameters.updateParams = 0

    u_ref, v_ref, w_ref, r_ref = mpc_step(mpcController)
    sendCanPacket([39, ("int16", u_ref), ("int16", v_ref), ("int16", w_ref), ("int16", r_ref)])
    time.sleep(mpcController.stepHorizon)


  print("MPC Thread stopped")


#Class for network controller
class ComHandler:
  def __init__(self, ip:str='0.0.0.0', port:int=6900, canifaceType:str='socketcan', canifaceName:str='can0') -> None:
    self.canifaceType  = canifaceType
    self.canifaceName  = canifaceName
    self.status    = {'Net': False, 'Can': False}
    self.uCstatus  = {'Reg': False, 'Sensor': False, '12Vman': False, '12Vthr': False, '5V': False}
    self.camStatus = {'Threads': False, 'S1': False, 'S2': False, 'Bottom': False, 'Manipulator': False}
    self.canFilters= [{'can_id': 0x80, 'can_mask': 0xE0, 'extended': False}]
    self.connectIp = ip
    self.connectPort = port
    self.servo = ServoPWM(pin=32, freq=50, startDT=7.5)
    self.canInit()
    self.camInit()
    self.netInit()
    self.heartBeat()
    self.i2cInit()
    self.mpcInit()



  def canInit(self):
    self.bus = can.Bus(interface = self.canifaceType, channel = self.canifaceName, receive_own_messages = False, fd = False)
    self.bus.set_filters(self.canFilters)
    self.status['Can'] = True
    self.notifier = can.Notifier(self.bus, [self.canCallback])

  def netInit(self):
    self.netHandler = Network(is_server = True, bind_addr = self.connectIp, port = self.connectPort)
    while self.netHandler.waiting_for_conn:
      time.sleep(1)
    self.toggleNet()

  def toggleNet(self):
    if self.status['Net']:
      self.status['Net'] = False
    else:
      self.netTrad = threading.Thread(name="Network_thread",target=netThread, daemon=True, args=(self.netHandler, self.netCallback, self.status))
      self.netTrad.start()

  def netCallback(self, data: bytes) -> None:
    functionsParsingDict  = {
      CAMERA: {TILT: self.servo.newAngle,
               START: self.camStart,
               STOP: self.camStop
               }
            }
    data:str = bytes.decode(data, 'utf-8')
    for message in data.split(json.dumps("*")):
      try:
        if message == json.dumps('heartbeat') or message == "":
          if message is None:
            message = ""
            continue
        else:
          message = json.loads(message)
          for item in message:

            if item[0] in canSendDict:
              if self.status['Can']:
                  msg = canSendDict[item[0]](item)
                  self.sendCanPacket(msg)
              else:
                self.sendTcpPacket("Error: Canbus not initialised")
            elif item[0] in functionsParsingDict:
              if item[1][0] in functionsParsingDict[item[0]]:
                print(f"function: {item[0]}, action: {item[1][0]}, with value: {item[1][1]} Activated")
                functionsParsingDict[item[0]][item[1][0]](item[1][1])
              else:
                print(f"function: {item[0]}, action: {item[1][0]}, with value: {item[1][1]} failed")
                #functionsParsingDict[item[0]](item[1])
            else: 
              self.sendTcpPacket(f'Error: canId: {item[0]} not mapped')                        
      except Exception as e:
            print(f'Feilkode i netCallback, feilmelding: {e}\n\t{message}')

  def sendTcpPacket(self, msg):
    self.msg = toJson(msg)
    if self.status['Net']:
      self.netHandler.send(self.msg)
      
  def sendCanPacket(self, tag):
    packet = packetBuild(tag)
    assert self.bus is not None
    try:
      self.bus.send(packet)
    except Exception as e:
      print(f'Feilkode i sendCanPacket, feilmelding: {e}\n\t{packet}')

  def canCallback(self, msg):
    if self.status['Can'] and self.status['Net']:
      self.bus.socket.settimeout(0)
      canID = msg.arbitration_id
      dataByte = msg.data
      if canID in canReciveDict:
            jsonDict = canReciveDict[canID](canID, dataByte, self.uCstatus)
      else:
            print(f"CanID: {canID} recived from ROV system not in parsing dict msg: {msg}")
            jsonDict = {"Error": f"CanID: {canID} recived from ROV system not in parsing dict with"}
      self.sendTcpPacket(jsonDict)
        
  def heartBeat(self):
    self.heartBeatThread = threading.Thread(name="hbThread",target=hbThread, daemon=True, args=(self.netHandler, self.sendCanPacket, self.status, self.uCstatus))
    self.heartBeatThread.start()
  
  def i2cInit(self):
    self.STTS75 = STTS75() 
    self.i2cThread = threading.Thread(name="i2cThread" ,target=i2cThread, daemon=True, args=(self.netHandler, self.STTS75, self.status))
    self.i2cThread.start()

  def mpcInit(self):
    self.mpcParams = MPCParameters()
    self.mpcController = MPCController(self.mpcParams)
    self.mpcThread = threading.Thread(name="mpcThread", target=mpcThread, deamon=True, args=(self.mpcParams, self.mpcController, self.status, self.sendCanPacket))
    self.mpcThread.start()


  #def PWM(self):
  #  self.servo = ServoPWM()
  #  self.PWMThread = threading.Thread(name="PWMThread",target=PWMThread, daemon=True, args=(self.servo))
  #  self.PWMThread.start()

  def camInit(self):
    #Gst.init([])
    self.stereo1Pipe = gstreamerPipe(pipeId="stereo1", port="5000")
    self.stereo1Thread = threading.Thread(target=self.stereo1Pipe.run)
    self.stereo1Thread.start()
    self.stereo2Pipe = gstreamerPipe(pipeId="stereo2", port="5001")
    self.stereo2Thread = threading.Thread(target=self.stereo2Pipe.run)
    self.stereo2Thread.start()
    self.bottomPipe = gstreamerPipe(pipeId="bottom", port="5002")
    self.bottomThread = threading.Thread(target=self.bottomPipe.run)
    self.bottomThread.start()
    self.manipulatorPipe = gstreamerPipe(pipeId="manipulator", port="5003")
    self.manipulatorThread = threading.Thread(target=self.manipulatorPipe.run)
    self.manipulatorThread.start()
    self.camStatus['Threads'] = True
    self.camStart('stereo1')
    self.camStart('stereo2')
    self.camStart('bottom')
    self.camStart('manipulator')

  def camStart(self, pipeId):
    if pipeId == 'stereo1':
      self.stereo1Pipe.runPipe()
      self.camStatus['S1'] = True
    elif pipeId == 'stereo2':
      self.stereo2Pipe.runPipe()
      self.camStatus['S2'] = True
    elif pipeId == 'bottom' and (self.camStatus['S1'] or self.camStatus['S2']): #freak bug where one of stereo cams must be running to start usb cams.
      self.bottomPipe.runPipe()
      self.camStatus['bottom'] = True
    elif pipeId == 'manipulator' and (self.camStatus['S1'] or self.camStatus['S2']): #freak bug where one of stereo cams must be running to start usb cams.
      self.manipulatorPipe.runPipe()
      self.camStatus['manipulator'] = True
    print(f"Camera: {pipeId} started")
    self.sendTcpPacket(f"Camera: {pipeId} started")
     
  def camStop(self, pipeId):
    if pipeId == 'stereo1':
      self.stereo1Pipe.stopPipe()
      self.camStatus['S1'] = False
    elif pipeId == 'stereo2':
      self.stereo2Pipe.stopPipe()
      self.camStatus['S2'] = False
    elif pipeId == 'bottom':
      self.bottomPipe.stopPipe()
      self.camStatus['bottom'] = False
    elif pipeId == 'manipulator':
      self.manipulatorPipe.stopPipe()
      self.camStatus['manipulator'] = False
    self.sendTcpPacket(f"Camera: {pipeId} stopped")

if __name__ == "__main__":
  c = ComHandler()
  while True:
    pass