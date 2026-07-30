set ffmpeg="c:\Program Files\ImageMagick\ffmpeg.exe"

set size=128

rem set filename=saturn_raw.mov 
rem set dirname=saturn_raw
rem set center_x=985
rem set center_y=510

set filename=jupiter_raw.mov 
set dirname=jupiter_raw
set center_x=1005
set center_y=530
set numframes=500

mkdir %dirname%
%ffmpeg% -i %filename% -frames:v %numframes% -filter:v "crop=%size%:%size%:%center_x%-%size%/2:%center_y%-%size%/2" %dirname%\%%08d.png

