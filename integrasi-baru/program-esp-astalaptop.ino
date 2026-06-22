#include <esp_now.h>
#include <WiFi.h>
#include <esp_wifi.h> 

// Harus IDENTIK dengan struct pengirim
typedef struct struct_pesan_laptop {
  char payload[120];
} struct_pesan_laptop;

struct_pesan_laptop paketAkhir;

void saatDataDiterima(const esp_now_recv_info_t *info, const uint8_t *incomingData, int len) {
  memcpy(&paketAkhir, incomingData, sizeof(paketAkhir));

  int rssi_jarak2 = (info != NULL && info->rx_ctrl != NULL) ? info->rx_ctrl->rssi : -120;

  // Forward ke laptop dengan tambah rssi_jarak2
  Serial.printf("DATA|%s|%d\n", paketAkhir.payload, rssi_jarak2);

  // Info analisa
  Serial.printf(">> [INFO] Jarak 2: %d dBm\n", rssi_jarak2);
}

void setup() {
  Serial.begin(115200);
  delay(1000); 

  WiFi.mode(WIFI_STA);
  WiFi.disconnect(); 
  delay(100);
  
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous(false);
  
  if (esp_now_init() != ESP_OK) return;
  
  esp_now_register_recv_cb((esp_now_recv_cb_t)saatDataDiterima);
  Serial.println("\n[ SERVER XAMPP READY - MENUNGGU KIRIMAN DARI JETSON S3 ]");
}

void loop() {
  // Mode Standby murni
}