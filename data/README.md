
# Persian-OtherLanguges  Machine Translation statistics

>In this repo we provide comperhensive statistics for exsiting Machine Translation dataset in Persian. At this stage we just represent statistics Persian-English datasets.

> We also prove Word-Level and Charachter-Level bar chart for each dataset which can be found in folderss dataset.

<br/><br/>
# Persian-English

<div align="center">

|       datasets      |   train   |  dev  |  test  |    all    |
|:-------------------:|:---------:|:-----:|:------:|:---------:|
|        Mizan        | 1,006,430 | 5,000 | 10,166 | 1,021,596 |
|  PEPC_Bidirectional |  175,442  | 5,000 | 19,494 |  199,936  |
| PEPC_Onedirectional |  138,005  | 5,000 | 15,334 |  158,339  |
|         TEP         |   72,748  | 5,000 |  8,084 |   85,832  |
|        TEP++        |  515,925  | 5,000 | 57,326 |  578,251  |
|    OpenSubtitles    | 1,000,000 | 2,000 |  2,000 | 1,004,000 |
|        Bible        |   51,329  | 5,000 |  5,704 |   62,033  |
|        Quran        | 1,013,756 | 5,000 | 10,240 | 1,028,996 |
|       ParsiNLU      | 1,617,271 | 2,138 | 48,123 | 1,667,532 |

</div>


<div align="center">
Word level <b>Persian-English</b>

|       dataset       | avg_fa | min_fa | max_fa | 92%_fa |   all_fa   | unique_fa | avg_en | min_en | max_en | 92%_en |   all_en   | unique_en |
|:-------------------:|:------:|:------:|:------:|:------:|:----------:|:---------:|:------:|:------:|:------:|:------:|:----------:|:---------:|
|        Bible        |   28   |    3   |   124  |   48   |  1,796,084 |   18,166  |   23   |    2   |   100  |   38   |  1,428,716 |   40,202  |
|        Mizan        |   13   |    1   |   232  |   26   | 13,464,236 |  131,751  |   13   |    0   |   226  |   26   | 13,360,397 |  259,182  |
|    OpenSubtitles    |   10   |    1   |  1,487 |   21   | 10,284,744 |  155,874  |    9   |    1   |   839  |   20   |  9,524,220 |  342,979  |
|       ParsiNLU      |   11   |    1   |  7,989 |   24   | 19,397,145 |  198,460  |   12   |    1   | 15,156 |   23   | 20,328,220 |  441,282  |
|  PEPC_Bidirectional |   20   |    7   |   178  |   35   |  4,163,011 |  169,637  |   21   |    7   |   153  |   36   |  4,354,619 |  142,792  |
| PEPC_Onedirectional |   22   |    7   |   178  |   37   |  3,539,183 |  158,707  |   21   |    7   |   153  |   36   |  3,359,635 |  138,489  |
|        Quran        |   29   |    1   |   373  |   61   | 30,235,077 |   28,380  |   33   |    1   |   772  |   74   | 34,227,828 |   92,976  |
|         TEP         |    8   |    1   |   37   |   14   |   716,113  |   22,710  |    7   |    1   |   33   |   14   |   684,242  |   36,634  |
|        TEP++        |    7   |    1   |   34   |   13   |  4,445,543 |   92,037  |    8   |    0   |   32   |   14   |  4,720,821 |   57,753  |

</div>


<br/><br/>

<div align="center">
Charachter level <b>Persian-English</b>

|       dataset       | avgc_fa | minc_fa | maxc_fa | avgc_en | minc_en | maxc_en |
|:-------------------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|        Bible        |   130   |    11   |   566   |   104   |    7    |   474   |
|        Mizan        |    62   |    1    |  1,242  |    59   |    1    |   986   |
|    OpenSubtitles    |    43   |    1    |  17,501 |    41   |    1    |  12,213 |
|       ParsiNLU      |    53   |    1    |  38,339 |    54   |    1    |  75,246 |
|  PEPC_Bidirectional |   105   |    23   |   870   |   127   |    15   |   868   |
| PEPC_Onedirectional |   112   |    23   |   870   |   124   |    15   |   868   |
|        Quran        |   135   |    3    |  1,725  |   148   |    2    |  3,312  |
|         TEP         |    35   |    1    |   156   |    33   |    1    |   144   |
|        TEP++        |    34   |    1    |   149   |    36   |    1    |   154   |

</div>


# ParsiNLU
Persian word distrobution             |  English word distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/ParsiNLU/eda/fa_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/ParsiNLU/eda/en_length_distrobution.png?raw=true)


Persian charachter distrobution             |  English charachter distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/ParsiNLU/eda/fa_ch_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/ParsiNLU/eda/en_ch_length_distrobution.png?raw=true)


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