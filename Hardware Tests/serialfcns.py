
import serial #import serial library

def wait_byte(ser,ack):

    ack_read = False
    val = ''
    while ack_read == False:
        v = ser.read()
        val = v.decode('ascii')
        if val == ack:
            ack_read = True
    return val