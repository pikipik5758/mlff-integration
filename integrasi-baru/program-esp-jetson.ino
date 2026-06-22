#include <esp_now.h>
#include <WiFi.h>
#include <esp_wifi.h>

// TARGET: MAC Address ESP WROOM-32 (Laptop)
uint8_t macLaptop[] = {0x0C, 0xB8, 0x15, 0xC3, 0xC8, 0x70};

// Struktur 1: Datang dari Gerbang
typedef struct struct_pesan_gerbang {
  char epc[25];
  int rssi_kosong;
} struct_pesan_gerbang;
struct_pesan_gerbang dataMasuk;

// HAPUS struct lama, ganti jadi:
typedef struct struct_pesan_laptop {
  char payload[120];  // string bebas, cukup untuk semua format
} struct_pesan_laptop;

struct_pesan_laptop dataKeluar;

void saatDataTerkirim(const uint8_t *mac_addr, esp_now_send_status_t status) { }

void saatDataDiterima(const esp_now_recv_info_t *info, const uint8_t *incomingData, int len) {
  memcpy(&dataMasuk, incomingData, sizeof(dataMasuk));
  
  // Tangkap RSSI Jarak 1 (Gerbang -> Jetson)
  int rssiAsli = (info != NULL && info->rx_ctrl != NULL) ? info->rx_ctrl->rssi : -120;
  
  // Kirim string ke port Serial (Tunggu dibaca program Python/ROS di Jetson)
  Serial.println("==============================================");
  Serial.printf("Tag ID : %s\n", dataMasuk.epc);
  Serial.printf("RSSI 1 : %d dBm\n", rssiAsli);
  Serial.println("==============================================\n");
}

void setup() {
  Serial.begin(115200);
  delay(2000); // Waktu inisialisasi ekstra untuk Native USB S3

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);

  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous(false);

  if (esp_now_init() != ESP_OK) return;

  esp_now_register_recv_cb((esp_now_recv_cb_t)saatDataDiterima);
  esp_now_register_send_cb((esp_now_send_cb_t)saatDataTerkirim);

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, macLaptop, 6);
  peerInfo.channel = 1;  
  peerInfo.encrypt = false;
  esp_now_add_peer(&peerInfo);
  
  Serial.println("\n[ RELAY S3 READY - SIAP DIHUBUNGKAN KE PYTHON JETSON ]");
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.length() > 0 && input.length() < 120) {
      input.toCharArray(dataKeluar.payload, 120);
      esp_now_send(macLaptop, (uint8_t *) &dataKeluar, sizeof(dataKeluar));
    }
  }
}