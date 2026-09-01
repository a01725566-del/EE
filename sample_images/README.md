# Put your traffic photo(s) here

`yolo_detector.py` and `run_experiment.py` expect an image at:

```
sample_images/traffic_scene_1.jpg
```

## Where to get a traffic scene photo

For the essay, a good image should look like something an Apollo Go
robotaxi's camera would actually see: a street-level view with a mix
of cars, pedestrians, and ideally a traffic light or crossing. Some
options, roughly from most to least recommended:

1. **A frame grabbed from a real Apollo Go ride-along video** (e.g. a
   YouTube video of someone riding in an Apollo Go robotaxi). Pause
   the video, screenshot the frame. This is the most directly relevant
   to the RQ, but remember to cite the video (title, channel,
   timestamp, URL, date accessed) in your EE's bibliography/appendix.
2. **A frame from an open self-driving dataset** (e.g. BDD100K,
   nuScenes, or Comma2k19 sample images) - these are made for exactly
   this kind of research and usually have clear licences to cite.
3. **A free-to-use stock photo of traffic** (e.g. from Wikimedia
   Commons, Pexels or Unsplash, filtered to Creative Commons /
   public-domain images) - cite the source and licence.
4. **A photo you took yourself** of a street - simplest for licensing
   (it's yours), but may not resemble Apollo Go's actual operating
   environment (check your EE's methodology/limitations section
   mentions this).

Whatever you use, keep a copy of the *original, unmodified* image
plus its source link somewhere safe - your EE will need to reference
it, and you may want to show it as Figure 1 (the "before" image)
next to the YOLO-labelled "after" images in results/.

You can use more than one image (e.g. `traffic_scene_1.jpg`,
`traffic_scene_2.jpg`) - just update `IMAGE_PATH` in `yolo_detector.py`
/ `run_experiment.py`, or duplicate `run_experiment.py` per image, if
your EE compares more than one scenario.
