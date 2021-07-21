import serial

ser = serial.Serial('com16', 115200)
print('开始测试')
while True:
    recv = ser.readline()
    recv = str(recv)
    yaw = int(recv[3:6])
    pitch = int(recv[7:10])
    # print(recv)
    print(yaw, pitch)
    if str(recv) == 'q':
        break
