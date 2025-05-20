Face Recognition
We provide ibug.face_detection and ibug.face_alignment in this repository. You can install directly from github repositories or by using compressed files.

Option 1. Install from github repositories
Git LFS, needed for downloading the pretrained weights that are larger than 100 MB.
You could install Homebrew and then install git-lfs without sudo priviledges.

Install ibug.face_detection
```
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
cd ..
```

Install ibug.face_alignment

```
git clone https://github.com/hhj1897/face_alignment.git
cd face_alignment
pip install -e .
cd ..
```

Option 2. Install by using compressed files
If you are experiencing over-quota issues for the above repositoies, you can download both packages ibug.face_detection and ibug.face_alignment, unzip the files, and then run pip install -e . to install each package.

Install ibug.face_detection
wget https://www.doc.ic.ac.uk/~pm4115/tracker/face_detection.zip -O ./face_detection.zip
unzip -o ./face_detection.zip -d ./
cd face_detection
pip install -e .
cd ..
Install ibug.face_alignment
wget https://www.doc.ic.ac.uk/~pm4115/tracker/face_alignment.zip -O ./face_alignment.zip
unzip -o ./face_alignment.zip -d ./
cd face_alignment
pip install -e .
cd ..
