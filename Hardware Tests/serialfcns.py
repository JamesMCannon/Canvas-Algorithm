
import serial #import serial library

def wait4byte(ser,ack):

    ack_read = False
    val = ''
    while ack_read == False:
        v = ser.read()
        val = v
        val = v.decode('utf-8')
        if val == ack:
            ack_read = True
    return val