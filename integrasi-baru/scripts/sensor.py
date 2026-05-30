# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import String
# import serial
# import time

# SERIAL_PORT = '/dev/ttyACM0'
# BAUD_RATE = 115200

# class SensorNode(Node):
#     def __init__(self):
#         super().__init__('rfid_sensor')
#         self.pub = self.create_publisher(String, '/rfid_data', 10)

#         try:
#             # Buat objek serial kosong dulu
#             self.ser = serial.Serial()
#             self.ser.port = SERIAL_PORT
#             self.ser.baudrate = BAUD_RATE
#             self.ser.timeout = 1

#             # KUNCI UTAMA: Matikan DTR dan RTS sebelum dibuka
#             self.ser.dtr = False
#             self.ser.rts = False

#             # Buka port
#             self.ser.open()
#             time.sleep(0.5)  # Beri waktu stabilisasi

#             self.get_logger().info(f"Terhubung ke ESP32-S3 di {SERIAL_PORT}")
#             self.get_logger().info("Menunggu data dari Gerbang...")

#         except Exception as e:
#             self.get_logger().error(f"Gagal buka serial: {e}")
#             self.ser = None
#             return

#         self.timer = self.create_timer(0.05, self.baca_serial)

#     def baca_serial(self):
#         if self.ser is None or not self.ser.is_open:
#             return

#         try:
#             if self.ser.in_waiting > 0:
#                 baris = self.ser.readline().decode('utf-8', errors='ignore').strip()

#                 if baris:
#                     self.get_logger().info(f"[RAW] {baris}")  # Debug: lihat semua data

#                     if baris.startswith('RFID_IN'):
#                         parts = baris.split(',')
#                         if len(parts) >= 2:
#                             epc  = parts[1]
#                             rssi = parts[2] if len(parts) >= 3 else 'N/A'
#                             self.get_logger().info(f"EPC: {epc} | RSSI: {rssi}")
#                             msg = String()
#                             msg.data = epc
#                             self.pub.publish(msg)

#         except Exception as e:
#             self.get_logger().warn(f"Error baca serial: {e}")

#     def destroy_node(self):
#         if self.ser and self.ser.is_open:
#             self.ser.close()
#         super().destroy_node()

# def main(args=None):
#     rclpy.init(args=args)
#     node = SensorNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()




#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import time

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE   = 115200

class SensorNode(Node):
    def __init__(self):
        super().__init__('rfid_sensor')
        self.pub = self.create_publisher(String, '/rfid_data', 10)

        # Buffer sementara untuk kumpulkan baris per blok
        self._epc_temp  = None
        self._rssi_temp = None

        try:
            self.ser = serial.Serial()
            self.ser.port     = SERIAL_PORT
            self.ser.baudrate = BAUD_RATE
            self.ser.timeout  = 1
            self.ser.dtr      = False
            self.ser.rts      = False
            self.ser.open()
            time.sleep(0.5)
            self.get_logger().info(f"Terhubung ke ESP32-S3 di {SERIAL_PORT}")
            self.get_logger().info("Menunggu data dari Gerbang...")
        except Exception as e:
            self.get_logger().error(f"Gagal buka serial: {e}")
            self.ser = None
            return

        self.timer = self.create_timer(0.05, self.baca_serial)

    def baca_serial(self):
        if self.ser is None or not self.ser.is_open:
            return

        try:
            if self.ser.in_waiting > 0:
                baris = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not baris:
                    return

                self.get_logger().info(f"[RAW] {baris}")

                # Tangkap EPC
                if baris.startswith("Tag ID"):
                    # Format: "Tag ID : E28069150000700B41525446"
                    parts = baris.split(":")
                    if len(parts) >= 2:
                        self._epc_temp = parts[-1].strip()

                # Tangkap RSSI
                elif baris.startswith("RSSI"):
                    # Format: "RSSI 1 : -42 dBm"
                    parts = baris.split(":")
                    if len(parts) >= 2:
                        self._rssi_temp = parts[-1].strip()  # "-42 dBm"

                # Garis pemisah → publish jika EPC sudah ada
                elif baris.startswith("==="):
                    if self._epc_temp:
                        rssi = self._rssi_temp if self._rssi_temp else "N/A"
                        self.get_logger().info(f"[PUBLISH] EPC: {self._epc_temp} | RSSI: {rssi}")
                        msg      = String()
                        msg.data = f"{self._epc_temp},{rssi}"
                        self.pub.publish(msg)
                        # Reset buffer
                        self._epc_temp  = None
                        self._rssi_temp = None

        except Exception as e:
            self.get_logger().warn(f"Error baca serial: {e}")

    def destroy_node(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()