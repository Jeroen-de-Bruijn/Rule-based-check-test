"""This module contains the business logic of the function.

Use the automation_context module to wrap your function in an Automate context helper
"""

import numpy as np

from pydantic import Field, SecretStr
from speckle_automate import (
    AutomateBase,
    AutomationContext,
    execute_automate_function,
)

from flatten import flatten_base

# import
from specklepy.api import operations
from specklepy.transports.server import ServerTransport

class FunctionInputs(AutomateBase):
    """These are function author defined values.

    Automate will make sure to supply them matching the types specified here.
    Please use the pydantic model schema to define your inputs:
    https://docs.pydantic.dev/latest/usage/models/
    """

    # an example how to use secret values
    # whisper_message: SecretStr = Field(title="This is a secret message")

    otherVersionId: str = Field(
        title="Other version ID",
        description="Specify the ID of the other version to compare with.",
    )
    
    # delete_objects: bool = Field(
    #    default=False,
    #    title="TEST Delete objects",
    #    description="Enable this option to delete the objects.",
    #)

def get_parameter_value(speckle_object):
    """
    Given a Speckle object, retrieve the value of parameter_name.
    Args:
        speckle_object: The Speckle object containing parameters.
    Returns:
        The value of parameter_name if found, or None if not present.
    """
    
    # Test
    parameter_name = "Area"

    try:
        # Assuming speckle_object is a dictionary or has a similar structure
        param_value = speckle_object.get(parameter_name)
        return param_value
    except KeyError:
        # parameter_name not found in the object
        return None


def get_average_point(speckle_objects):
    filtered_objects = []
    average_points = []

    for count, speckle_object in enumerate(speckle_objects):
        mesh_list = speckle_object["displayValue"]
        
        if not len(mesh_list) >= 1:
            # No mesh found so go to the next item in the loop
            continue
        
        # Add object with mesh to list
        filtered_objects.append(speckle_object)

        # New empty list
        vertices_tuples = []
        
        # Loop through each mesh
        for m in mesh_list:
            vertices_list = m["vertices"]
            # Convert the flat list into a list of tuples
            temp_vertices_tuples = [(vertices_list[i], vertices_list[i + 1], vertices_list[i + 2]) for i in range(0, len(vertices_list), 3)]
            # Extend (so it remains one list) the list with these points 
            vertices_tuples.extend(temp_vertices_tuples)
        
        # Convert to NumPy array
        vertices_array = np.array(vertices_tuples)
        # Get average point
        average_p = np.mean(vertices_array, axis=0)
        # Add to list
        average_points.append(average_p)
    
    # Return lists
    return filtered_objects, average_points


def automate_function(
    automate_context: AutomationContext,
    function_inputs: FunctionInputs,
) -> None:
    """This is an example Speckle Automate function.

    Args:
        automate_context: A context helper object, that carries relevant information
            about the runtime context of this function.
            It gives access to the Speckle project data, that triggered this run.
            It also has convenience methods attach result data to the Speckle model.
        function_inputs: An instance object matching the defined schema.
    """
    print("test function 01")
    
    # The context provides a convenient way, to receive the triggering version
    version_root_object = automate_context.receive_version()

    print("test function 02")
    
    # Get other version from the same project (a different project doesn't seem to work, probably by design)
    project_id = automate_context.automation_run_data.project_id
    # version_id = "488ab9d83b"
    version_id = function_inputs.otherVersionId
    the_speckle_client = automate_context.speckle_client
    other_commit = the_speckle_client.commit.get(project_id, version_id)
    
    print("other_commit:")
    print(other_commit)
    
    if not other_commit.referencedObject:
        raise ValueError("The commit has no referencedObject, cannot receive it.")
    
    other_server_transport = ServerTransport(stream_id=project_id, client=the_speckle_client)
    other_root_object = operations.receive(other_commit.referencedObject, other_server_transport)

    print("test function 03")

    # Get all walls so we can check and compare their types
    find_this_speckle_type = "Objects.BuiltElements.Wall:Objects.BuiltElements.Revit.RevitWall"
    
    objects_match = [
        b
        for b in flatten_base(version_root_object)
        if b.speckle_type == find_this_speckle_type
    ]
    count = len(objects_match)
    
    other_objects_match = [
        b
        for b in flatten_base(other_root_object)
        if b.speckle_type == find_this_speckle_type
    ]
    other_count = len(other_objects_match)
    
    # Check mesh and get average point
    objects_match_clean, list_points = get_average_point(objects_match)
    other_objects_match_clean, other_list_points = get_average_point(other_objects_match)

    print("test function 04 point lists created")

    # Rename lists
    list_A = list_points
    list_B = other_list_points
    
    # Convert lists to NumPy arrays for efficient computation
    array_A = np.array(list_A)
    array_B = np.array(list_B)

    # Initialize an array to keep track of whether points in list_B have been used
    used_indices = np.zeros(len(list_B), dtype=bool)

    # Initialize lists to store the closest points
    closest_points = []
    closest_walls = []
    match_distance = []
    matching_walls = []

    # Iterate over each point in list_A
    for point_A in array_A:
        # Calculate distances from point_A to all unused points in list_B
        distances = np.linalg.norm(point_A - array_B[~used_indices], axis=1)
        
        # Find the index of the closest unused point in list_B
        closest_index = np.argmin(distances)
        
        # Mark the closest point as used
        used_indices[closest_index] = True
        
        # Append the closest point to the result
        closest_points.append(list_B[closest_index])
        closest_walls.append(other_objects_match_clean[closest_index])
        match_distance.append(round(distances[closest_index], 2))

    # matches = {"distance": X, "wall1": other_object, "wall2": different_object}

    print("test function 05 closest points calculated")
    
    successful = []
    failed = []

    # Loop through all walls
    for i, wall2 in enumerate(closest_walls):
        wall1 = objects_match_clean[i]
        type1 = wall1["type"]
        type2 = wall2["type"]
        combined_data = wall1.id, type1, type2, match_distance[i]
        
        if type1 == type2:
            successful.append(combined_data)
        else:
            failed.append(combined_data)
    
    count_success = len(successful)
    count_fail = len(failed)

    if count_success >= 1:
        wall_ids = [combined_data[0] for combined_data in successful]
        types1 = [combined_data[1] for combined_data in successful]
        types2 = [combined_data[2] for combined_data in successful]
        matched_dist = [combined_data[3] for combined_data in successful]
        
        automate_context.attach_info_to_objects(
            category = "Type check ok",
            object_ids = wall_ids,
            message = f"Wall1 types: {types1}\nWall2 types: {types2}\nDistance: {matched_dist}"
        )
    
    if count_fail >= 1:
        wall_ids = [combined_data[0] for combined_data in failed]
        types1 = [combined_data[1] for combined_data in failed]
        types2 = [combined_data[2] for combined_data in failed]
        matched_dist = [combined_data[3] for combined_data in failed]
        
        automate_context.attach_warning_to_objects(
            category = "Type check FAILED",
            object_ids = wall_ids,
            message = f"Wall1 types: {types1}\nWall2 types: {types2}\nDistance: {matched_dist}"
        )

    print("test function code completed")

    if count_fail == 0 and count_success == 0:
        automate_context.mark_run_failed("Automation run failed. Something went wrong. It looks like no walls were found")
    else:
        automate_context.mark_run_success(
            "Automation completed: "
            f"{count_fail} objects have non matching wall types! It's advised to review them.\n"
            f"{count_success} objects have matching wall types."
        )

        # set the automation context view, to the original model / version view
        # to show the offending objects
        automate_context.set_context_view()


def automate_function_without_inputs(automate_context: AutomationContext) -> None:
    """A function example without inputs.

    If your function does not need any input variables,
     besides what the automation context provides,
     the inputs argument can be omitted.
    """
    pass


# make sure to call the function with the executor
if __name__ == "__main__":
    # NOTE: always pass in the automate function by its reference, do not invoke it!

    # pass in the function reference with the inputs schema to the executor
    execute_automate_function(automate_function, FunctionInputs)

    # if the function has no arguments, the executor can handle it like so
    # execute_automate_function(automate_function_without_inputs)
