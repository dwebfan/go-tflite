1. copy your original preview folder to one directory such as `/tmp/lomo_preview`, and remove unnecessary images like 
```
cd /tmp/local_preview/
find . -name "*.mov" -name "*.png" -name "*_75_0.jpg" -name "*_200_0.jpg" | xargs rm
```
2. go -mod=vendor build

 NOTE: if you are running build at 32 bit raspberry pi, please use command `CGO_LDFLAGS="-latomic" go -mod=vendor build`

3. test with one sample directory, such as
`./label_image_dir /tmp/local_preview/alice/Photos/preview/2020/01/01`
4. browse all images under ./label_images, and see if labels are added correctly
5. run same command for left images
`./label_image_dir /tmp/local_preview`
