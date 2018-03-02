**Program ini berjalan menggunakan Python Interpreter versi 3**

Berikut adalah cara menggunakan program ini :
1. Sebelum menjalankan program, perlu diperhatikan untuk menginstall requirements dengan menggunakan pip
    ```sh
    Command : pip install -r requirements.txt
    ```
2. Untuk menampilkan visualisasi dapat menjalankan Visualization.py
    ```sh
    Command : python3 Visualization.py atau python Visualization.py
    ```
3. Untuk mencari nilai g paling optimum dan menampilkan visualisasinya, dapat menjalankan FindG.py
    ```sh
    Command : python3 FindG.py atau python FindG.py
    ```
    Program tersebut akan mengoutputkan tabel csv yang berisi nilai g dan akurasi yang didapat per iterasi pada folder outputs dan namafile : findingG.csv dan menampilkan grafik antara nilai g dan akurasi

4. Untuk melakukan Pengujian terhadap data test, dapat menjalankan file SolveDataTest.py
    ```sh
    Command : python3 SolveDataTest.py atau python SolveDataTest.py
    ```
    Lalu akan diminta untuk memasukkan nilai g yang akan digunakan, jika tidak menginputkan atau input error, maka akan menggunakan nilai g default yaitu hasil dari pencarian g yang paling optimum.
