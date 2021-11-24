import pickle        
import base64
import csv





with open('/home/thien/Desktop/Human-Falling-Detect-Tracks/Data/Coffee_room_new-set(labelXscrw).pkl', 'a', encoding='utf8') as csv_file:
    wr = csv.writer(csv_file, delimiter='|')
    pickle_bytes = pickle.dumps(obj)            # unsafe to write
    b64_bytes = base64.b64encode(pickle_bytes)  # safe to write but still bytes
    b64_str = b64_bytes.decode('utf8')          # safe and in utf8
    wr.writerow(['col1', 'col2', b64_str])