


q3

Text (Kod): Chrome'un diskten RAM'e aktarılan derlenmiş C++ komutları.


Data (Veri): Chrome'un sürüm numarası veya varsayılan tema seçimi gibi baştan belli olan ve process boyunca değişmeyen sabit değerleri.


Heap (Öbek): YouTube'da sayfayı aşağı kaydırdıkça RAM'e sürekli eklenen dinamik video kapak resimleri ve yorumlar.


Stack (Yığın): Arama çubuğuna "youtube.com" yazıp Enter'a basıldığında o kısacık anda kullanılan geçici bağlantı parametreleri.

q4

new: Kodu çalıştırmak için enter.


admitted: Kod diskten okundu, RAM'e yerleştirildi.


ready: Kod RAM'de sırasını (CPU'yu) bekliyor.


scheduler dispatch: İşletim sistemi "sıra sende" deyip kodu işlemciye aldı.


running: İşlemci kodları (hesaplamaları) aktif olarak çalıştırıyor.


interrupt: İşletim sistemi "süren bitti" deyip süreci kesti, tekrar sıraya yolladı.


I/O or event wait: Kod büyük bir dosya (örneğin diskten veri) beklemek için işlemciyi bıraktı.


waiting: İşlemci serbest bırakıldı, kod sadece diskin işini bitirmesini bekliyor.


I/O or event completion: Disk veriyi getirdi, süreç uyandı ama direkt işlemciye değil, tekrar sıraya girdi.


exit: Kodun işi hatasız bitti, kapanış komutu verildi.


terminated: İşlem tamamen kapandı, RAM'den silindi.

Dördüncü soru olan PCB (Process Control Block) için de bu tarz tek