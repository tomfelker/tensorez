set ffmpeg="c:\Program Files\ImageMagick\ffmpeg.exe"

set size=512

rem set filename=saturn_raw.mov 
rem set dirname=saturn_raw
rem set center_x=985
rem set center_y=510

rem set filename=jupiter_raw.mov 
rem set dirname=jupiter_raw
rem set center_x=1005
rem set center_y=530
rem set numframes=500

rem set filename=jupiter_mvi_6906.mov 
rem set dirname=jupiter_mvi_6906
rem set center_x=965
rem set center_y=550
rem set numframes=100

set filename=saturn_bright_mvi_6902.mov 
set  dirname=saturn_bright_mvi_6902
set center_x=995
set center_y=570
set numframes=100

mkdir %dirname%
%ffmpeg% -i %filename% -frames:v %numframes% -filter:v "crop=%size%:%size%:%center_x%-%size%/2:%center_y%-%size%/2" %dirname%\%%08d.png

