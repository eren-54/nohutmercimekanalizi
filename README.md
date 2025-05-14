# nohutmercimekanalizi
Bu proje, nohut ve mercimek tanelerini görüntü işleme ve makine öğrenmesi teknikleri kullanarak ayırt etmeyi amaçlamaktadır. Özellikle tarımsal ürünlerin otomatik olarak sınıflandırılması gereken durumlar için kullanılabilir.

İlk olarak, görsellerin arka planı rembg kütüphanesi ile kaldırılmıştır. Ardından her bir görüntüden renk (ortalama R, G, B), doku (GLCM matrisinden elde edilen kontrast, homojenlik gibi) ve boyut (alan, çevre) gibi özellikler çıkarılmıştır. Bu veriler daha sonra normalize edilerek bir CSV dosyasında saklanmıştır.

Makine öğrenmesi kısmında, KNN (K-Nearest Neighbors) algoritması kullanılarak sınıflandırma modeli eğitilmiştir. Modelin eğitimi sırasında her sınıf için dengeli sayıda örnek kullanılmış, model joblib ile diske kaydedilmiş ve doğruluk oranı test seti ile ölçülmüştür.

Projeye ayrıca görsel yükleyerek canlı tahmin yapılabilen bir Tkinter arayüzü de entegre edilmiştir. Kullanıcı bir görsel seçtiğinde sistem önce arka planı temizler, ardından aynı özellik çıkarım ve tahmin sürecini çalıştırarak sonucu kullanıcıya gösterir.
