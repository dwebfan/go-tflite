package main

import (
	"bufio"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	_ "image/jpeg"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/disintegration/imaging"
	"github.com/golang/freetype"
	"github.com/mattn/go-tflite"
	"github.com/nfnt/resize"
	"golang.org/x/image/font/gofont/gomono"
)

type result struct {
	Score float64
	Index int
}

func loadLabels(filename string) ([]string, error) {
	labels := []string{}
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	return labels, nil
}

func main() {
	rootDir := ""
	outDir := "label_images"
	if len(os.Args) == 2 {
		rootDir = os.Args[1]
	} else if len(os.Args) > 1 {
		rootDir = os.Args[1]
		outDir = os.Args[2]
	} else {
		log.Fatal(errors.New("usage: label_image_dir [source image directory] ([analysised image directory])"))
	}
	models := []*tflite.Model{}
	labels := [][]string{}
	modelName := []string{"mobilenet", "i_v4"}
	modelPath := []string{"../label_image/mobilenet_quant_v1_224.tflite", "inception_v4_quant.tflite"}
	labelPath := []string{"labels.txt", "labels.txt"}

	for i := 0; i < len(labelPath); i++ {
		label, err := loadLabels(labelPath[i])
		if err != nil {
			log.Printf("load label file %s\n", labelPath[i])
			log.Fatal(err)
		}
		labels = append(labels, label)

		model := tflite.NewModelFromFile(modelPath[i])
		if model == nil {
			log.Fatal("cannot load model")
		}
		defer model.Delete()
		models = append(models, model)
	}

	options := tflite.NewInterpreterOptions()
	options.SetNumThread(8)
	options.SetErrorReporter(func(msg string, user_data interface{}) {
		fmt.Println(msg)
	}, nil)
	defer options.Delete()

	err := filepath.Walk(rootDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		texts := []string{}
		for i := 0; i < len(models); i++ {
			results, err := calc(models[i], options, path)
			if err != nil {
				return err
			}
			for j := 0; j < len(results); j++ {
				label := labels[i][results[j].Index]
				rule, _ := rules.Find(label)

				// discard labels that don't met the threshold
				if float32(results[j].Score) < rule.Threshold {
					continue
				}
				texts = append(texts, fmt.Sprintf("%s - %s(%s): %.2f", modelName[i], label, rule.Label, results[j].Score))
				if j > 5 {
					break
				}
			}
		}
		if len(texts) == 0 {
			return nil
		}

		parts := strings.Split(path, string(os.PathSeparator))
		outpath := info.Name()
		if len(parts) > 7 {
			// assume directories is like .../alice/Photos/preview/2020/01/01
			out := filepath.Join(outDir,
				parts[len(parts)-7], parts[len(parts)-6], parts[len(parts)-5],
				parts[len(parts)-4], parts[len(parts)-3], parts[len(parts)-2])
			if err := os.MkdirAll(out, 0755); err != nil {
				return err
			}
			outpath = filepath.Join(out, info.Name())
		}
		return outputImage(path, outpath, texts)
	})
	if err != nil {
		log.Fatal(err)
	}
}

func calc(model *tflite.Model, options *tflite.InterpreterOptions, fp string) ([]result, error) {
	f, err := os.Open(fp)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}

	interpreter := tflite.NewInterpreter(model, options)
	if interpreter == nil {
		return nil, errors.New("cannot create interpreter")
	}
	defer interpreter.Delete()

	status := interpreter.AllocateTensors()
	if status != tflite.OK {
		return nil, errors.New("allocate failed")
	}

	input := interpreter.GetInputTensor(0)
	wantedHeight := input.Dim(1)
	wantedWidth := input.Dim(2)
	wantedChannels := input.Dim(3)
	wantedType := input.Type()

	resized := resize.Resize(uint(wantedWidth), uint(wantedHeight), img, resize.NearestNeighbor)
	bounds := resized.Bounds()
	dx, dy := bounds.Dx(), bounds.Dy()

	if wantedType == tflite.UInt8 {
		bb := make([]byte, dx*dy*wantedChannels)
		for y := 0; y < dy; y++ {
			for x := 0; x < dx; x++ {
				col := resized.At(x, y)
				r, g, b, _ := col.RGBA()
				bb[(y*dx+x)*3+0] = byte(float64(r) / 255.0)
				bb[(y*dx+x)*3+1] = byte(float64(g) / 255.0)
				bb[(y*dx+x)*3+2] = byte(float64(b) / 255.0)
			}
		}
		input.CopyFromBuffer(bb)
	} else {
		return nil, fmt.Errorf("%d is not wanted type", wantedType)
	}

	status = interpreter.Invoke()
	if status != tflite.OK {
		return nil, errors.New("invoke failed")
	}

	output := interpreter.GetOutputTensor(0)
	outputSize := output.Dim(output.NumDims() - 1)
	b := make([]byte, outputSize)
	status = output.CopyToBuffer(&b[0])
	if status != tflite.OK {
		return nil, errors.New("output failed")
	}
	results := []result{}
	for i := 0; i < outputSize; i++ {
		score := float64(b[i]) / 255.0
		if score < 0.2 {
			continue
		}
		results = append(results, result{Score: score, Index: i})
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	return results, nil
}

func convertValue(value uint32) float32 {
	return (float32(value>>8) - float32(127.5)) / float32(127.5)
}

func outputImage(in, out string, labels []string) error {
	fmt.Printf("save [%s] into %s\n", strings.Join(labels, ","), out)
	f, err := os.Open(in)
	if err != nil {
		return err
	}
	defer f.Close()
	bimg, err := jpeg.Decode(f)
	if err != nil {
		return err
	}
	fontSize := 15
	y := fontSize
	// use hard coded width to display full label
	//width := bimg.Bounds().Dx()
	width := 400
	height := bimg.Bounds().Dy()
	img := addBanner(bimg, width, height, (len(labels)+1)*fontSize)
	for _, s := range labels {
		opt := Options{
			TTF:             gomono.TTF,
			Foreground:      color.RGBA{0xff, 0x00, 0x00, 0xff},
			Background:      color.RGBA{0xcc, 0xcc, 0xcc, 0xff},
			BackgroundImage: img,
		}
		gen, err := NewImageGenerator(opt)
		if err != nil {
			return err
		}
		img, err = gen.NewPlaceholderPos(s, width, height, 2, y, float64(fontSize))
		if err != nil {
			return err
		}
		y += fontSize
	}

	o, err := os.Create(out)
	if err != nil {
		return err
	}
	defer o.Close()
	return jpeg.Encode(o, img, nil)
}

func addBanner(backImg image.Image, width, height, bannerHeight int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, width, height))

	c := freetype.NewContext()
	c.SetDPI(dpi)
	c.SetSrc(image.NewUniform(color.RGBA{0, 0, 0, 0}))
	c.SetDst(img)
	c.SetClip(img.Bounds())

	// draw background image
	bgimg := imaging.Fill(backImg, width, height, imaging.Center, imaging.Lanczos)
	draw.Draw(img, img.Bounds(), bgimg, image.ZP, draw.Src)
	draw.Draw(img, img.Bounds(),
		image.NewRGBA(image.Rect(0, 0, width, bannerHeight)),
		image.ZP, draw.Src)

	return img
}
