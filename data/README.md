
# Persian-OtherLanguges  Machine Translation statistics

>In this repo we provide comperhensive statistics for exsiting Machine Translation dataset in Persian. At this stage we just represent statistics Persian-English datasets.

> We also prove Word-Level and Charachter-Level bar chart for each dataset which can be found in folderss dataset.

<br/><br/>
# Persian-English

<div align="center">

|       datasets      |  train  |  dev |  test |   all   |
|:-------------------:|:-------:|:----:|:-----:|:-------:|
|        Mizan        | 1006430 | 5000 | 10166 | 1021596 |
|  PEPC_Bidirectional |  175442 | 5000 | 19494 |  199936 |
| PEPC_Onedirectional |  138005 | 5000 | 15334 |  158339 |
|         TEP         |  72748  | 5000 |  8084 |  85832  |
|        TEP++        |  515925 | 5000 | 57326 |  578251 |
|    OpenSubtitles    | 1000000 | 2000 |  2000 | 1004000 |

</div>


<div align="center">
Word level <b>Persian-English</b>

|       dataset       | avg_fa | min_fa | max_fa | 92%_fa |  all_fa  | unique_fa | avg_en | min_en | max_en | 92%_en |  all_en  | unique_en |
|:-------------------:|:------:|:------:|:------:|:------:|:--------:|:---------:|:------:|:------:|:------:|:------:|:--------:|:---------:|
|        Mizan        |   13   |    1   |   232  |   26   | 13464236 |   131751  |   13   |    0   |   226  |   26   | 13360397 |   259182  |
|    OpenSubtitles    |   10   |    1   |  1487  |   21   | 10284744 |   155874  |    9   |    1   |   839  |   20   |  9524220 |   342979  |
|  PEPC_Bidirectional |   20   |    7   |   178  |   35   |  4163011 |   169637  |   21   |    7   |   153  |   36   |  4354619 |   142792  |
| PEPC_Onedirectional |   22   |    7   |   178  |   37   |  3539183 |   158707  |   21   |    7   |   153  |   36   |  3359635 |   138489  |
|         TEP         |    8   |    1   |   37   |   14   |  716113  |   22710   |    7   |    1   |   33   |   14   |  684242  |   36634   |
|        TEP++        |    7   |    1   |   34   |   13   |  4445543 |   92037   |    8   |    0   |   32   |   14   |  4720821 |   57753   |

</div>


<br/><br/>

<div align="center">
Charachter level <b>Persian-English</b>

|       dataset       | avgc_fa | minc_fa | maxc_fa | avgc_en | minc_en | maxc_en |
|:-------------------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|        Mizan        |    62   |    1    |   1242  |    59   |    1    |   986   |
|    OpenSubtitles    |    43   |    1    |  17501  |    41   |    1    |  12213  |
|  PEPC_Bidirectional |   105   |    23   |   870   |   127   |    15   |   868   |
| PEPC_Onedirectional |   112   |    23   |   870   |   124   |    15   |   868   |
|         TEP         |    35   |    1    |   156   |    33   |    1    |   144   |
|        TEP++        |    34   |    1    |   149   |    36   |    1    |   154   |

</div>



# TEP
Persian word distrobution             |  English word distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/TEP/eda/fa_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/TEP/eda/en_length_distrobution.png?raw=true)


Persian charachter distrobution             |  English charachter distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/TEP/eda/fa_ch_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/TEP/eda/en_ch_length_distrobution.png?raw=true)


# TEP++
Persian word distrobution             |  English word distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/TEP++/eda/fa_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/TEP++/eda/en_length_distrobution.png?raw=true)


Persian charachter distrobution             |  English charachter distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/TEP++/eda/fa_ch_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/TEP++/eda/en_ch_length_distrobution.png?raw=true)


# PEPC_Bidirectional
Persian word distrobution             |  English word distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/PEPC_Bidirectional/eda/fa_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/PEPC_Bidirectional/eda/en_length_distrobution.png?raw=true)


Persian charachter distrobution             |  English charachter distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/PEPC_Bidirectional/eda/fa_ch_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/PEPC_Bidirectional/eda/en_ch_length_distrobution.png?raw=true)



# PEPC_Onedirectional
Persian word distrobution             |  English word distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/PEPC_Onedirectional/eda/fa_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/PEPC_Onedirectional/eda/en_length_distrobution.png?raw=true)


Persian charachter distrobution             |  English charachter distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/PEPC_Onedirectional/eda/fa_ch_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/PEPC_Onedirectional/eda/en_ch_length_distrobution.png?raw=true)


# Mizan
Persian word distrobution             |  English word distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/Mizan/eda/fa_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/Mizan/eda/en_length_distrobution.png?raw=true)


Persian charachter distrobution             |  English charachter distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/Mizan/eda/fa_ch_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/Mizan/eda/en_ch_length_distrobution.png?raw=true)


# OpenSubtitles
Persian word distrobution             |  English word distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/OpenSubtitles/eda/fa_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/OpenSubtitles/eda/en_length_distrobution.png?raw=true)


Persian charachter distrobution             |  English charachter distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/OpenSubtitles/eda/fa_ch_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/OpenSubtitles/eda/en_ch_length_distrobution.png?raw=true)


# Quran
Persian word distrobution             |  English word distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/Quran/eda/fa_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/Quran/eda/en_length_distrobution.png?raw=true)


Persian charachter distrobution             |  English charachter distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/Quran/eda/fa_ch_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/Quran/eda/en_ch_length_distrobution.png?raw=true)


# Bible
Persian word distrobution             |  English word distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/Bible/eda/fa_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/Bible/eda/en_length_distrobution.png?raw=true)


Persian charachter distrobution             |  English charachter distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/Bible/eda/fa_ch_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/Bible/eda/en_ch_length_distrobution.png?raw=true)