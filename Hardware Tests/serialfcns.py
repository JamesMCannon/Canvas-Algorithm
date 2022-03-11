
import serial #import serial library

def wait4byte(ser,ack,is_ascii=True):

    ack_read = False
    val = ''
    while ack_read == False:
        v = ser.read()
        if is_ascii:
            val = v.decode('ascii')
        else:
            val=v
        if val == ack:
            ack_read = True
    return val