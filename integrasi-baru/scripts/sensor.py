# VERSI TAMBAH TIMESTAMP & DELAY
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import time
from datetime import datetime

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE   = 115200

class SensorNode(Node):
    def __init__(self):
        super().__init__('rfid_sensor')
        self.pub = self.create_publisher(String, '/rfid_data', 10)

        self.create_subscription(String, '/kirim_ke_esp', self.callback_kirim_esp, 10)

        self._epc_temp  = None
        self._rssi_temp = None

        # ↓ TAMBAH: untuk hitung delay antar pesan
        self._waktu_terakhir = None

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

                if baris.startswith("Tag ID"):
                    parts = baris.split(":")
                    if len(parts) >= 2:
                        self._epc_temp = parts[-1].strip()

                elif baris.startswith("RSSI"):
                    parts = baris.split(":")
                    if len(parts) >= 2:
                        self._rssi_temp = parts[-1].strip()

                elif baris.startswith("==="):
                    if self._epc_temp:
                        rssi = self._rssi_temp if self._rssi_temp else "N/A"

                        # ↓ TAMBAH: hitung timestamp & delay
                        waktu_sekarang = time.time()
                        t1 = round(waktu_sekarang, 3)  # untuk hitung delay e2e di penerima

                        if self._waktu_terakhir is not None:
                            delay_antar_tag = round(waktu_sekarang - self._waktu_terakhir, 3)
                        else:
                            delay_antar_tag = 0.0

                        self._waktu_terakhir = waktu_sekarang

                        self.get_logger().info(
                            f"[PUBLISH] EPC: {self._epc_temp} | RSSI: {rssi} | "
                            f"T1: {t1} | Delay antar tag: {delay_antar_tag}s"
                        )

                        msg      = String()
                        msg.data = f"{self._epc_temp},{rssi},{t1},{delay_antar_tag}"
                        self.pub.publish(msg)

                        self._epc_temp  = None
                        self._rssi_temp = None

        except Exception as e:
            self.get_logger().warn(f"Error baca serial: {e}")

    def callback_kirim_esp(self, msg):
        if self.ser is None or not self.ser.is_open:
            return
        try:
            pesan = msg.data
            if not pesan.endswith('\n'):
                pesan += '\n'
            self.ser.write(pesan.encode('utf-8'))
            self.get_logger().info(f"[KIRIM ESP] {pesan.strip()}")
        except Exception as e:
            self.get_logger().warn(f"Error kirim ke ESP: {e}")

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