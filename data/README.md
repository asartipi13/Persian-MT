
# Persian-OtherLanguges  Machine Translation statistics

>In this repo we provide comperhensive statistics for exsiting Machine Translation dataset in Persian. At this stage we just represent statistics Persian-English datasets.

> We also prove Word-Level and Charachter-Level bar chart for each dataset which can be found in folderss dataset.

<br/><br/>
# Persian-English

<div align="center">

|       datasets      |   train   |   dev  |   test  |    all    |
|:-------------------:|:---------:|:------:|:-------:|:---------:|
|        Mizan        | 1006430.0 | 5000.0 | 10166.0 | 1021596.0 |
|  PEPC_Bidirectional |  175442.0 | 5000.0 | 19494.0 |  199936.0 |
| PEPC_Onedirectional |  138005.0 | 5000.0 | 15334.0 |  158339.0 |
|         TEP         |  72748.0  | 5000.0 |  8084.0 |  85832.0  |
|        TEP++        |  515925.0 | 5000.0 | 57326.0 |  578251.0 |
|    OpenSubtitles    | 1000000.0 | 2000.0 |  2000.0 | 1004000.0 |
</div>


<div align="center">
Word level <b>Persian-English</b>

|       dataset       | avg_fa | min_fa | max_fa | 92%_fa |  all_fa | unique_fa | avg_en | min_en | max_en | 92%_en |  all_en | unique_en |
|:-------------------:|:------:|:------:|:------:|:------:|:-------:|:---------:|:------:|:------:|:------:|:------:|:-------:|:---------:|
|  PEPC_Bidirectional |   21   |    6   |   194  |   36   | 4295971 |   140512  |   21   |    7   |   153  |   37   | 4397290 |   134726  |
| PEPC_Onedirectional |   23   |    6   |   194  |   38   | 3650404 |   132567  |   21   |    7   |   153  |   36   | 3391029 |   131110  |
|         TEP         |    7   |    1   |   32   |   14   |  682786 |   36793   |    8   |    1   |   37   |   14   |  716605 |   22615   |


</div>


<br/><br/>

<div align="center">
Charachter level <b>Persian-English</b>


|       dataset       | avgc_fa | minc_fa | maxc_fa | avgc_en | minc_en | maxc_en |
|:-------------------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|  PEPC_Bidirectional |   105   |    23   |   870   |   129   |    17   |   870   |
| PEPC_Onedirectional |   112   |    23   |   870   |   126   |    17   |   870   |
|         TEP         |    33   |    1    |   144   |    37   |    3    |   158   |

</div>



# TEP
Persian word distrobution             |  English word distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/TEP/eda/fa_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/TEP/eda/en_length_distrobution.png?raw=true)


Persian charachter distrobution             |  English charachter distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/TEP/eda/fa_ch_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/TEP/eda/en_ch_length_distrobution.png?raw=true)


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

