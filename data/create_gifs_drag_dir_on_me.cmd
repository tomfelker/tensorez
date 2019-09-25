set imagemagick_dir="c:\Program Files\ImageMagick\"

cd %1
%imagemagick_dir%\magick.exe convert -delay 0 -loop 0 "%%08d.png[1-40]" raw_40.gif



