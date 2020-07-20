import argparse


parser = argparse.ArgumentParser(description="Give the groundtruth test images and infered images ")
parser.add_argument("--groundtruth", type=str, default="../../testset_boundingboxes/",
                    help="suffix for files")
parser.add_argument("--infer", type=str, default="data/processed/test20190617160600_bicubic/seedlings/", help="data path ")
args=parser.parse_args()


html_head = """<html>
<head> <title> Test results from a model </title></head>
<body>

"""

html_tail = """
</body>
</html>
"""

filenames=[]
with open('data/interim/datasets/test_seedling_paths')as file:
    for line in file:
        filename=line.replace('\n','').split('/')[-1]
        filenames.append(filename)

print(filenames)

with open(args.infer+'test_result.html','w')as file:
    file.write(html_head)
    for filename in filenames:
        groundtruth = args.groundtruth +filename
        text='<p>left infered, right ground truth: {}</p>'.format(filename)
        # output html in the infered test image directory.
        infered_img ='<p><img src="{}" alt="infered" > '.format(filename)
        groundtruth_img = '<img src="{}" alt="infered" > </p>'.format(groundtruth)

        file.write(text)
        file.write(infered_img)
        file.write(groundtruth_img)
    file.write(html_tail)