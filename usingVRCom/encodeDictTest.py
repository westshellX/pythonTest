"""
The following example shows how to create an AIS message using a dictionary of values.

The full specification of the AIVDM/AIVDO protocol is out of the scope of this example.
For a good overview of the AIVDM/AIVDO Sentence Layer please refer to this project: https://gpsd.gitlab.io/gpsd/AIVDM.html#_aivdmaivdo_sentence_layer

But you should keep the following things in mind:

- AIS messages are part of a two layer protocol
- the outer layer is the NMEA 0183 data exchange format
- the actual AIS message is part of the NMEA 0183’s 82-character payload
- because some AIS messages are larger than 82 characters they need to be split across several fragments
- there are 27 different types of AIS messages which differ in terms of fields

Now to the actual encoding of messages: It is possible to encode a dictionary of values into an AIS message.
To do so, you need some values that you want to encode. The keys need to match the interface of the actual message.
You can call `.fields()` on any message class, to get glimpse on the available fields for each message type.
Unknown keys in the dict are simply omitted by pyais. Most keys have default values and do not need to
be passed explicitly. Only the keys `type` and `mmsi` are always required

For the following example, let's assume that we want to create a type 1 AIS message.
"""
# Required imports
from pyais.encode import encode_dict
from pyais.messages import MessageType1

'''   msg_type = bit_field(6, int, default=1)
    repeat = bit_field(2, int, default=0)
    mmsi = bit_field(30, int, from_converter=from_mmsi, to_converter=to_mmsi)
    status = bit_field(4, int, default=0, converter=NavigationStatus.from_value)
    turn = bit_field(8, int, default=0, signed=True)
    speed = bit_field(10, int, from_converter=from_speed, to_converter=to_speed, default=0)
    accuracy = bit_field(1, int, default=0)
    lon = bit_field(28, int, from_converter=from_lat_lon, to_converter=to_lat_lon, default=0, signed=True)
    lat = bit_field(27, int, from_converter=from_lat_lon, to_converter=to_lat_lon, default=0, signed=True)
    course = bit_field(12, int, from_converter=from_course, to_converter=to_course, default=0)
    heading = bit_field(9, int, default=0)
    second = bit_field(6, int, default=0)
    maneuver = bit_field(2, int, default=0, from_converter=ManeuverIndicator.from_value,
                         to_converter=ManeuverIndicator.from_value)
    spare = bit_field(3, int, default=0)
    raim = bit_field(1, bool, default=0)
    radio = bit_field(19, int, default=0)
'''

def encodeDict(speed,course,heading,lat,lon,mmsi,type,isOS=False)->str:
# This statement tells us which fields can be set for messages of type 1
    print("Hello DictTest!")
#    print(MessageType1.fields())

# A dictionary of fields that we want to encode
# Note that you can pass many more fields for type 1 messages, but we let pyais
# use default values for those keys
    speedInt=10
    data = {
        'speed':speedInt,
        'course': course,
        'heading':heading,
        'lat': lat,
        'lon': lon,
        'mmsi': mmsi,
        'type': type
    }

# This creates an encoded AIS message
# Note, that `encode_dict` returns always a list of fragments.
# This is done, because you may never know if a message fits into the 82 character
# size limit of payloads
    talker_id="AIVDM"
    if(isOS):
        talker_id="AIVDO"

# You can also change the NMEA fields like the radio channel:
    messageList=encode_dict(data,talker_id,radio_channel="B")
    messageStr='{}\r\n'.format(messageList[0])
    return messageStr
