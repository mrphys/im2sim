Thoughts:
model seems ok for now. Don’t modify anything but keep in mind modification when designing the framework.
repeated scaling is annoying. Save as file in local workspace and move to main code folder when done - then it’ll become the same as pca
build out more of an api than what it currently is. 
Let the user specify what elements they want to insert in the callback/data generator. The base is just a skeleton then. 
Do they want to add a custom augmentation or input something specific? Swap it out with the default one.
Do they want to use some sort of preset protocol? That should be available as an option 
Make the fixed dataclasses mutable dicts instead and let the user specify the contents of the dictionary - also include presets for this 
Each built-in function can come with a desciption of the stuff it needs. Perhaps we move away from passing the full dict and instead we map certain dict keys to the function? This needs to be considered more carefully. 
The model and losses need to accept some subset of the generator output. The user needs to make sure that the losses have the required keys. 
Since there are so many different variables that are needed at different stages, we need to generate a report showing what parameters we’re missing. So the workflow will be to configure your model/losses etc and run the report generator to see whether everything is as expected. That way you won't have to keep checking for errors. 
We need to calculate the sim output losses on an interpolated basis. This, along with mapping a dict to the losses will remove the need for registration  and give the model a better understanding of the relationship between geometry and physics
Physics-based losses - these should be easy to integrate once everything else is in place
More augmentations
mesh-based augmentations can be applied at runtime for segmentation problems but not for im2sim problems 
Image augmentations including noise, contrast and translation/rotation can be applied for any type of problem


Datasets
- loading vtus is a lot slower than loading .pt files so:
    - On the first epoch, load in the data using pyvista, perform the preprocessing and cache
        - concatenate the node type to the node features 
        - write all the utils you need based on the edges and node type
    - On subsequent epochs, access the cache and load the data
    - perform augmentation on the preprocessed data 
