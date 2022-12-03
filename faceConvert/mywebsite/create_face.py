import cv2
import mediapipe as mp
import bpy
import math
import numpy as np

Face_index =\
[(10,109,108),(109,69,108),(67,69,109),(67,104,69),(103,67,104),
(103,104,68),(103,68,54),(54,68,71),(54,71,21),(21,71,139),
(21,139,162),(162,139,34),(34,162,127),(127,34,227),(227,127,234),
(234,227,137),(234,137,93),(93,137,177),(177,93,132),(132,177,58),
(58,177,215),(215,58,172),(172,215,138),(138,172,136),(138,136,135),
(135,136,150),(150,135,169),(150,169,149),(149,169,170),(170,149,140),
(140,149,176),(140,176,171),(171,176,148),(148,171,152),(152,171,175),
(175,152,396),(396,377,152),(377,396,400),(400,396,369),(369,400,378),
(378,369,395),(395,378,394),(394,378,379),(394,379,364),(364,379,365),
(364,365,367),(367,365,397),(397,367,435),(435,397,288),(288,435,401),
(401,288,361),(361,401,323),(323,401,366),(366,323,454),(366,454,447),
(447,454,356),(356,447,264),(264,356,389),(389,264,368),(368,389,251),
(368,251,301),(301,251,284),(284,301,298),(298,284,332),(333,332,298),
(332,297,333),(333,332,297),(297,333,299),(299,297,338),(338,299,337),
(337,338,10),(337,151,10),(10,151,108),(108,151,107),(107,108,66),
(66,108,69),(69,66,105),(105,104,69),(105,104,63),(63,104,68),
(68,63,70),(68,70,71),(71,70,156),(156,71,139),(139,156,143),
(139,143,34),(34,143,116),(116,34,227),(227,116,123),(123,227,137),
(137,123,147),(147,137,177),(147,177,213),(213,177,215),(215,213,192),
(192,215,138),(138,192,135),(135,192,214),(214,135,210),(210,135,169),
(169,210,170),(170,210,211),(211,170,32),(32,170,140),(140,32,208),
(32,140,208),(208,140,171),(171,208,175),(175,208,199),(199,428,175),
(175,396,428),(428,396,369),(369,428,262),(262,369,395),(395,262,431),
(431,395,430),(430,395,394),(394,430,364),(364,430,434),(434,364,416),
(416,364,367),(367,416,435),(435,433,416),(433,435,401),(401,433,376),
(376,401,366),(366,376,352),(366,352,447),(447,345,352),(447,345,264),
(264,345,372),(372,264,368),(368,372,383),(383,301,368),(383,301,300),
(300,301,383),(300,301,298),(298,300,293),(298,293,333),(333,293,334),
(334,333,299),(299,334,296),(296,299,337),(337,296,336),(336,337,151),
(151,336,9),(9,151,107),(107,9,55),(55,107,65),(65,107,66),
(65,66,52),(52,66,105),(105,52,53),(53,63,105),(63,53,46),
(46,63,70),(70,46,124),(124,70,156),(156,124,35),(35,156,143),
(35,143,111),(111,143,116),(116,111,117),(117,116,123),(123,117,50),
(50,123,187),(187,123,147),(147,187,192),(192,147,213),(187,192,207),
(192,207,214),(214,207,216),(216,214,212),(212,214,210),(210,212,202),
(202,210,211),(211,202,204),(204,211,194),(194,211,32),(32,194,201),
(32,201,208),(208,201,200),(200,208,199),(199,200,428),(428,200,421),
(421,428,262),(262,421,418),(418,262,431),(431,418,424),(424,431,422),
(431,422,430),(430,422,432),(432,430,434),(432,434,436),(436,434,427),
(434,427,416),(416,427,411),(411,416,376),(376,433,416),(411,376,352),
(280,411,352),(352,280,346),(346,352,345),(345,346,340),(340,345,372),
(340,372,265),(265,372,383),(383,265,353),(353,383,300),(300,353,276),
(276,300,293),(293,276,283),(283,293,334),(334,283,282),(282,334,296),
(296,282,295),(295,296,336),(336,295,285),(285,336,9),(9,285,8),
(9,8,55),(55,8,193),(193,55,221),(221,55,222),(222,55,65),
(65,222,52),(52,222,223),(223,52,224),(224,52,53),(53,224,225),
(53,46,225),(225,46,113),(113,46,124),(124,113,226),(226,124,35),
(35,226,31),(31,35,111),(31,111,117),(117,31,228),(228,117,118),
(118,117,50),(118,50,101),(101,50,205),(205,50,187),(187,205,207),
(207,205,206),(206,207,216),(216,206,92),(92,216,186),(186,216,212),
(212,57,186),(212,57,43),(43,212,202),(202,43,106),(106,202,204),
(204,106,182),(182,204,194),(194,182,201),(201,182,83),(83,201,200),
(83,200,18),(18,200,313),(313,200,421),(313,421,406),(406,421,418),
(406,418,424),(424,406,335),(335,424,422),(422,335,273),(273,422,432),
(273,432,287),(287,432,410),(410,432,436),(436,410,322),(322,436,426),
(426,436,427),(427,426,425),(425,427,411),(411,425,280),(280,425,330),
(330,280,347),(347,280,346),(346,347,448),(448,346,261),(261,346,340),
(340,261,265),(265,261,446),(446,265,353),(353,446,342),(342,353,276),
(445,342,276),(276,445,283),(283,445,444),(444,283,282),(282,444,443),
(443,442,282),(282,442,295),(295,442,285),(285,442,441),(441,285,417),
(8,417,285),(8,417,168),(168,8,193),(193,168,122),(122,193,245),
(245,193,189),(189,193,221),(221,189,56),(56,221,222),(222,56,28),
(222,223,28),(28,223,27),(27,223,29),(29,223,224),(29,224,30),
(30,224,225),(225,30,247),(247,225,113),(113,247,130),(130,113,226),
(226,130,25),(25,226,31),(31,25,228),(228,25,110),(110,228,229),
(229,228,118),(118,229,119),(118,119,101),(101,119,100),(100,101,36),
(36,101,205),(205,36,206),(206,36,203),(203,206,165),(165,206,92),
(92,165,39),(39,92,40),(40,92,186),(186,40,185),(185,186,57),
(185,57,61),(61,57,146),(146,57,43),(43,146,91),(91,43,106),
(106,91,182),(182,91,181),(181,182,84),(84,182,83),(83,84,18),
(84,17,18),(18,17,314),(314,18,313),(313,314,406),(406,314,405),
(405,406,321),(321,406,335),(335,321,273),(273,321,375),(375,273,287),
(287,375,291),(291,287,409),(409,287,410),(410,409,270),(270,410,322),
(322,270,269),(269,322,391),(391,322,426),(426,391,423),(423,426,266),
(266,426,425),(425,266,330),(330,266,329),(329,330,348),(348,330,347),
(348,347,449),(449,347,448),(448,449,339),(448,339,255),(255,448,261),
(261,255,446),(446,255,359),(359,446,342),(342,359,467),(467,342,445),
(467,445,260),(260,445,444),(444,260,259),(259,444,443),(443,259,257),
(257,258,443),(443,258,442),(442,258,286),(286,442,441),(441,286,413),
(413,441,417,),(417,413,465),(465,417,351),(351,417,168),(168,351,6),
(168,6,122),(122,6,196),(196,122,188),(188,122,245),(245,188,114),
(114,245,128),(128,244,245),(244,245,189),(189,244,190),(190,189,56),
(56,190,157),(157,56,158),(158,56,28),(28,158,159),(159,28,27),
(27,159,29),(159,29,160),(29,30,160),(160,30,161),(161,30,247),
(247,246,161),(246,247,33),(33,247,130),(130,33,25),(33,25,7),
(7,33,246),(246,7,163),(163,246,161),(161,163,144),(144,161,160),
(160,144,145),(145,160,159),(159,145,153),(153,159,158),(158,153,154),
(158,154,157),(157,154,155),(155,157,173),(173,155,133),(157,173,190),
(190,173,243),(173,133,243),(243,190,244),(244,243,233),(233,243,112),
(244,233,128),(133,243,112),(155,133,112),(112,155,26),(26,154,155),
(26,22,154),(154,153,22),(22,153,23),(153,23,145),(145,23,144),
(144,23,24),(24,144,110),(110,144,163),(163,7,110),(110,7,25),
(110,24,229),(229,24,230),(230,24,23),(23,230,231),(23,231,22),
(22,231,232),(232,22,26),(26,232,112),(232,112,233),(233,232,128),
(232,128,121),(121,232,231),(231,121,120),(120,231,230),(230,120,119),
(229,230,119),(119,120,100),(100,120,47),(47,120,121),(121,47,114),
(121,114,128),(188,196,174),(174,188,114),(114,174,217),(217,47,114),
(47,217,126),(126,47,100),(100,126,142),(142,100,36),(36,142,129),
(129,142,209),(209,142,126),(126,209,198),(198,126,217),(217,198,174),
(174,236,198),(196,236,3),(3,196,197),(197,3,195),(197,196,6),
(174,196,236),(3,195,51),(236,3,51),(51,236,134),(134,236,198),
(198,134,131),(131,198,49),(49,198,209),(49,209,129),(129,49,102),
(102,49,48),(48,102,64),(64,102,129),(49,48,131),(131,48,115),
(115,131,220),(220,131,134),(134,220,45),(45,134,51),(51,5,45),
(51,5,195),(5,4,45),(4,45,1),(1,45,44),(44,45,220),
(220,44,237),(237,218,220),(220,218,115),(115,218,219),(219,115,48),
(48,219,64),(64,219,235),(64,235,240),(240,64,98),(98,64,129),
(129,98,203),(203,98,165),(165,167,98),(98,97,167),(98,240,97),
(97,99,240),(240,99,75),(75,240,235),(75,59,235),(235,59,219),
(36,129,203),(165,167,39),(167,39,37),(37,167,164),(164,37,0),
(164,0,267),(164,267,393),(393,267,269),(393,269,391),(97,167,164),
(97,164,2),(2,164,326),(326,164,393),(393,326,327),(393,391,327),
(327,391,423),(423,327,358),(358,423,266),(358,266,371),(371,266,329),
(6,351,419),(419,6,197),(197,419,248),(248,197,195),(195,248,281),
(281,195,5),(5,281,275),(275,5,4),(4,275,1),(1,275,274),
(351,465,412),(351,412,419),(419,412,399),(419,399,456),(456,419,248),
(248,456,281),(281,456,363),(281,363,275),(275,363,440),(275,440,274),
(274,440,457),(464,465,357),(465,357,343),(343,465,412),(412,343,399),
(399,343,437),(437,399,420),(420,399,456),(456,420,363),(420,363,360),
(363,360,440),(360,440,344),(440,344,438),(440,438,457),(464,463,453),
(464,453,357),(463,341,453),(341,453,452),(453,452,357),(357,452,350),
(357,350,343),(343,350,277),(343,277,437),(437,277,355),(437,355,420),
(420,355,429),(429,420,279),(420,360,279),(360,279,278),(360,344,278),
(344,278,439),(344,438,439),(438,439,392),(438,392,309),(309,392,290),
(250,309,290),(458,250,309),(458,250,462),(462,458,461),(461,462,370),
(354,461,370),(370,354,94),(94,354,19),(19,94,125),(125,94,141),
(141,125,241),(241,141,242),(242,241,238),(238,242,20),(238,20,79),
(219,166,218),(218,166,79),(79,218,239),(239,218,237),(237,239,241),
(241,237,44),(44,241,125),(125,44,19),(19,44,1),(1,19,274),
(274,19,354),(354,274,461),(461,274,457),(457,461,459),(457,459,438),
(459,438,309),(309,459,458),(461,458,459),(239,241,238),(238,239,79),
(250,290,328),(328,250,462),(462,328,326),(462,326,370),(370,326,2),
(370,2,94),(94,2,141),(141,2,97),(97,141,242),(242,97,99),
(99,242,20),(99,20,60),(20,60,79),(79,60,166),(166,60,75),
(60,99,75),(75,59,166),(166,59,219),(413,286,414),(414,413,464),
(465,464,413),(464,463,414),(463,414,398),(463,362,398),(463,362,341),
(398,362,382),(362,382,341),(286,414,384),(414,384,398),(398,384,382),
(384,382,381),(382,381,256),(382,256,341),(341,256,452),(286,258,385),
(286,384,385),(384,385,381),(381,385,380),(380,381,252),(252,381,256),
(256,252,452),(452,252,451),(452,451,350),(350,451,349),(350,349,277),
(277,349,329),(329,277,355),(355,329,371),(371,355,429),(371,429,358),
(358,429,279),(279,331,358),(278,279,331),(331,278,294),(294,278,439),
(439,294,455),(455,439,289),(289,392,439),(289,392,305),(289,455,305),
(392,290,305),(290,305,328),(328,326,460),(326,460,327),(328,305,460),
(305,455,460),(455,294,460),(460,294,327),(294,327,358),(358,331,294),
(258,385,386),(258,257,386),(386,385,380),(380,386,374),(380,374,253),
(380,253,252),(252,253,451),(451,253,450),(450,451,349),(450,349,348),
(349,348,329),(257,259,386),(386,259,387),(387,386,374),(374,387,373),
(374,373,253),(253,373,254),(254,253,450),(450,254,449),(449,450,348),
(259,387,260),(260,387,388),(388,387,373),(373,388,390),(373,390,339),
(373,254,339),(339,254,449),(260,388,467),(467,388,466),(388,466,390),
(466,390,249),(390,249,339),(466,467,263),(263,466,249),(249,263,255),
(339,249,255),(255,263,359),(467,263,359),

(61,185,76),(76,185,184),(184,185,40),(40,184,74),(40,74,73),
(73,40,39),(39,73,72),(72,39,37),(37,72,0),(11,72,0),
(0,11,302),(267,0,302),(302,267,269),(269,302,303),(269,303,270),
(303,304,270),(270,304,408),(270,408,409),(409,408,306),(306,409,291),

(61,76,146),(146,76,77),(146,77,91),(91,77,90),(90,91,180),
(91,180,181),(181,180,85),(85,181,84),(85,84,16),(16,84,17),
(16,17,314),(314,16,315),(315,314,405),(405,315,404),(404,405,321),
(321,404,320),(320,321,307),(321,307,375),(375,307,306),(306,291,375),

(76,62,183),(184,183,76),(184,183,42),(42,184,74),(74,42,41),
(74,41,73),(73,41,38),(38,73,72),(72,38,12),(12,11,72),
(11,302,12),(12,302,268),(268,302,303),(303,268,271),(303,304,271),
(271,304,272),(304,272,408),(408,272,407),(407,408,306),(407,292,306),

(76,62,77),(62,77,96),(77,96,90),(96,89,90),(90,89,179),
(179,90,180),(180,179,86),(180,86,85),(85,86,15),(15,85,16),
(15,16,315),(15,315,316),(316,315,404),(404,316,403),(404,403,320),
(320,403,319),(319,320,325),(325,320,307),(307,325,292),(307,292,306),

(62,183,191),(191,183,80),(183,80,42),(42,80,81),(81,42,41),
(41,81,82),(82,41,38),(38,82,13),(38,12,13),(12,13,268),
(268,13,312),(312,268,271),(271,312,311),(271,272,311),(311,272,310),
(272,310,407),(407,310,415),(415,407,292),

(62,96,95),(95,96,88),(88,96,89),(88,89,179),(179,178,88),
(178,179,87),(87,179,86),(87,86,14),(14,15,86),(14,15,316),
(316,14,317),(317,316,403),(403,402,317),(402,403,318),(318,403,319),
(318,319,325),(325,318,324),(324,325,292),

(62,78,191),(62,78,95),(78,191,95),(191,95,80),(80,95,88),
(80,88,81),(81,88,178),(81,178,82),(82,178,87),(87,82,13),(87,13,14),
(13,14,317),(13,317,312),(312,317,402),(402,312,311),(311,402,318),
(311,318,310),(310,318,324),(324,310,415),(415,324,308),(415,308,292),(324,308,292),]

OUTLINE_POINTS_1 = [10,109,67,103,54,21,162,127,234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454,356,389,251,284,332,297,338]    #36개
OUTLINE_POINTS_2 = [151,108,69,104,68,71,139,34,227,137,177,215,138,135,169,170,140,171,175,396,369,395,394,364,367,435,401,366,447,264,368,301,298,333,299,337]  #36개
OUTLINE_POINTS_3 = [9,107,66,105,63,70,156,143,116,123,147,213,192,214,210,211,32,208,199,428,262,431,430,434,416,433,376,352,345,372,383,300,293,334,296,336]    #36개
OUTLINE_POINTS_4 = [8,55,65,52,53,46,124,35,111,117,50,187,207,216,212,202,204,194,201,200,421,418,424,422,432,436,427,411,280,346,340,265,353,276,283,282,295,285] #38개

RIGHT_OUTLINE_POINT = [128,121,120,119,118,101,100,47,114,205,36,206,203]
LEFT_OUTLINE_POINT = [357,350,349,348,347,330,329,277,343,425,266,426,423]

#눈
RIGHT_EYE_POINT_1 = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]   #16개
RIGHT_EYE_POINT_2 = [130,25,110,24,23,22,26,112,243,190,56,28,27,29,30,247]   #16개
RIGHT_EYE_POINT_3 = [226,31,228,229,230,231,232,233,244,189,221,222,223,224,225,113]   #16개

LEFT_EYE_POINT_1 = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]   #16개
LEFT_EYE_POINT_2 = [463,341,256,252,253,254,339,255,359,467,260,259,257,258,286,414]   #16개
LEFT_EYE_POINT_3 = [464,453,452,451,450,449,448,261,446,342,445,444,443,442,441,413]   #16개

#입술
MOUSE_POINT_1 = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]  #20개
MOUSE_POINT_2 = [62,96,89,179,86,15,316,403,319,325,292,407,272,271,268,12,38,41,42,183]  #20개
MOUSE_POINT_3 = [76,77,90,180,85,16,315,404,320,307,306,408,304,303,302,11,72,73,74,184]  #20개
MOUSE_POINT_4 = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]  #20개
MOUSE_POINT_5 = [57,43,106,182,83,18,313,406,335,273,287,410,322,391,393,164,167,165,92,186] #20개

NOSE_POINT_1 = [168,193,245,188,174,217,126,142,129,98,97,2,326,327,358,371,355,437,399,412,465,417]
MIDDLE_NOSE_LINE = [6,197,195,5,4,1,19,94]
RIGHT_NOSE_POINT = [122,196,3,51,45,44,125,141, 236,134,220,237,241,242, 198,131,115,218,239,238,20,79, 209,49,48,219,166,60,99, 75,240,59,235,64,102]
LEFT_NOSE_POINT = [351,419,248,281,275,274,354,370, 456,363,440,457,461,462, 420,360,344,438,459,458,250,309, 429,279,278,439,392,290,328, 305,460,289,455,294,331]

#밑에는 중복

NOSTRILL_POINT = [79,166,75,60,20,238, 309,392,305,290,250,458,459]

RIGHT_IRIS = [468,469,470,471,472]
LEFT_IRIS = [473,474,475,476,477]

#Face Mesh
# Face_line_index1 = [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
#                     152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338]
Face_line_index1 =  [10, 82, 57, 76, 45, 15, 135, 100, 201, 68, 105, 49, 145, 109, 123, 122, 149, 121,
                    125, 323, 346, 324, 325, 311, 343, 252, 307, 271, 394, 302, 335, 218, 248, 278, 260, 284]
Face_line_index2 = [331, 640, 641, 642, 643, 644,444,443,379,645,353,352,646,649,648,344,647,342,
                    2,1,338,7,340,339,337,15,14,336,41,106,105,335,334,333,332,330]

Combine_index = [(867,10), (866,284),(866,260),(866,278),(866,248),
                (866,218),(641,335),(641,302),(577,394),(863,271),
                (550,307),(550,252),(861,343),(862,311),(862,325),
                (543,324),(860,346),(537,323),

                (537,125),(859,121),(1163,149),(860,122),(1162,123),
                (1162,109),(1161,145),(867,49),(867,105),(1158,68),
                (892,201),(955,100),(955,135),(1154,15),(1153,45),
                (1152,76),(1151,57),(1150,82)
                ]

Mediate_index = [(314,335),(231,302),(387,394),(312,271),(347,307),(375,252),(313,343),
		        (111,145),(182,49),(150,105),(110,68),(194,201),(28,100),(112,135)]

print(len(Face_line_index1), len(Face_line_index2))

class createFaceMesh:
    def __init__(self, file_name):
        self.file_name = file_name
        ff = np.fromfile(self.file_name, np.uint8)
        self.image = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)
        # self.image = cv2.imread(self.file_name)
        self.rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
            self.result = face_mesh.process(self.rgb_image)

        self.expansion_ratio = None

    def delete_all_object(self):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
    def chooes_object(self,object_name):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_pattern(pattern=object_name)
        obj = bpy.context.selected_objects[0]
        obj.select_set(state=True)
        bpy.context.view_layer.objects.active = obj

    def create_hair(self):
        self.import_hair('back01', './static/hair/back/back01.obj', is_resize= True)
        self.move_origine_to_other_object_vertex('Visage', 729, 'back01', 709271)

        self.import_hair('left01', './static/hair/left/left01.obj', is_resize= True)
        self.move_origine_to_other_object_vertex('Visage', 757, 'left01', 232891)

        self.import_hair('right01', './static/hair/right/right01.obj', is_resize= True)
        self.move_origine_to_other_object_vertex('Visage', 1057, 'right01', 232829)

        self.import_hair('front01', './static/hair/front/front01.obj', is_resize= True)
        self.move_origine_to_other_object_vertex('Visage', 742, 'front01', 2851281)

    def import_head(self):
        # bpy.ops.import_scene.obj(filepath="./static/headpart.obj")
        bpy.ops.import_scene.obj(filepath="./static/object/headpart5.obj")
        for obj in bpy.context.selected_objects:
            obj.name = "headpart"

    #test
    def import_hair(self, object_name, path, is_resize=False):
        bpy.ops.import_scene.obj(filepath=path)
        for obj in bpy.context.selected_objects:
            obj.name = object_name

        if self.expansion_ratio != None and is_resize:
            bpy.data.objects[object_name].scale = self.expansion_ratio


    def rotate_objects(self):
        obj = bpy.data.objects['headpart']
        list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]

        start_2 = list[Face_line_index2[0]]
        end_2 = list[Face_line_index2[18]]

        obj = bpy.data.objects['Visage']
        list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]

        start_1 = list[Face_line_index1[0]]
        end_1 = list[Face_line_index1[18]]

        print(start_2, end_2, start_1, end_1)

        lx1, ly1, lz1 = start_1[0] - end_1[0], start_1[1] - end_1[1], start_1[2] - end_1[2]
        lx2, ly2, lz2 = start_2[0] - end_2[0], start_2[1] - end_2[1], start_2[2] - end_2[2]

        diff_1 = math.atan2(lx1,lz1)*180/math.pi - math.atan2(lx2,lz2)*180/math.pi
        diff_2 = math.atan2(ly1,lz1)*180/math.pi - math.atan2(ly2,lz2)*180/math.pi

        print(diff_1, diff_2)

        self.chooes_object('Visage')
        bpy.context.object.rotation_euler[0] += math.radians(diff_2)
        bpy.context.object.rotation_euler[1] += math.radians(diff_1)


    def resize_head(self, landmarks_point):
        self.rotate_objects()
        obj = bpy.data.objects['headpart']
        list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
        line_list = []
        for i in Face_line_index2:
            line_list.append(list[i])

        h_max_x = max(ls[0] for ls in line_list)
        h_max_y = max(ls[1] for ls in line_list)
        h_max_z = max(ls[2] for ls in line_list)

        h_min_x = min(ls[0] for ls in line_list)
        h_min_y = min(ls[1] for ls in line_list)
        h_min_z = min(ls[2] for ls in line_list)

        h_length_x = h_max_x - h_min_x
        h_length_y = h_max_y - h_min_y
        h_length_z = h_max_z - h_min_z

        headpart_cursor = list[128]
        # print(h_length_x, h_length_y, h_length_z)

        obj = bpy.data.objects['Visage']
        list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
        line_list = []
        for i in Face_line_index1:
            line_list.append(list[i])

        v_max_x = max(ls[0] for ls in line_list)
        v_max_y = max(ls[1] for ls in line_list)
        v_max_z = max(ls[2] for ls in line_list)

        v_min_x = min(ls[0] for ls in line_list)
        v_min_y = min(ls[1] for ls in line_list)
        v_min_z = min(ls[2] for ls in line_list)

        v_length_x = v_max_x - v_min_x
        v_length_y = v_max_y - v_min_y
        v_length_z = v_max_z - v_min_z

        visage_cursor = line_list[18]
        # print(v_length_x, v_length_y, v_length_z)



        # print(v_length_x/h_length_x, v_length_y/h_length_y, v_length_z/h_length_z)
        print("비율: ",(v_length_x / h_length_x, v_length_z / h_length_z, v_length_y / h_length_y))

        # bpy.data.objects['headpart'].scale = (h_length_x/v_length_x, h_length_y/v_length_y, h_length_z/v_length_z)
        # bpy.data.objects['headpart'].scale = (v_length_x/h_length_x, v_length_y/h_length_y, v_length_z/h_length_z)
        self.expansion_ratio = (v_length_x / h_length_x, v_length_z / h_length_z, v_length_y / h_length_y)
        bpy.data.objects['headpart'].scale = self.expansion_ratio
        # #테스트용
        # bpy.data.objects['Visage'].scale = (h_length_x / v_length_x, h_length_z / v_length_z, h_length_y / v_length_y)

    def move_origine_to_vertex(self, object_name, vertex_index, to_center = False):
        self.chooes_object(object_name)
        obj = bpy.context.view_layer.objects.active
        list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
        cursor_pos = list[vertex_index]
        bpy.context.scene.cursor.location = (cursor_pos[0], cursor_pos[1], cursor_pos[2])
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
        if to_center:
            bpy.context.object.location = (0, 0, 0)

    # 오브젝트1과 오브젝트2를 커서를 사용해서 붙이는 방법
    def move_origine_to_other_object_vertex(self, base_name, base_vertex_index, des_name, des_vertex_index):
        #오브젝트 2 커서 설정
        self.chooes_object(des_name)
        obj = bpy.context.view_layer.objects.active
        list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
        cursor_pos = list[des_vertex_index]
        bpy.context.scene.cursor.location = (cursor_pos[0], cursor_pos[1], cursor_pos[2])
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

        #오브젝트 1 좌표 설정
        self.chooes_object(base_name)
        obj = bpy.context.view_layer.objects.active
        list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
        cursor_pos = list[base_vertex_index]
        bpy.context.scene.cursor.location = (cursor_pos[0], cursor_pos[1], cursor_pos[2])

        #연결
        self.chooes_object(des_name)
        bpy.context.object.location = (cursor_pos[0], cursor_pos[1], cursor_pos[2])

    def combine_head(self):

        # self.move_origine_to_vertex('headpart', Face_line_index2[18], to_center= True)
        # self.move_origine_to_vertex('visage', Face_line_index1[18], to_center= True)

        self.move_origine_to_other_object_vertex('headpart', Face_line_index2[18], 'visage', Face_line_index1[18])

        self.objects_join(["Visage", "headpart"])
        self.move_origin_center('Visage')


        #합치기(연결)
        obj = bpy.context.view_layer.objects.active
        list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
        bpy.ops.object.mode_set(mode='EDIT')

        for i, order in enumerate(Combine_index):

            a = order[0]
            b = order[1]

            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.mode_set(mode='OBJECT')

            obj.data.vertices[a].select = True
            obj.data.vertices[b].select = True

            bpy.ops.object.mode_set(mode='EDIT')

            bpy.ops.mesh.merge(type= 'CENTER')

        bpy.ops.object.mode_set(mode='OBJECT')


        self.mediate_face()
        self.shade_smooth("Visage")


    #얼굴, 두상 점 연결
    def mediate_face(self):
        obj = bpy.data.objects[-1]
        obj = bpy.context.view_layer.objects.active
        list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
        last_dot = len(list)

        #좌표 조정
        bpy.ops.object.mode_set(mode='EDIT')
        for m, n in Mediate_index:
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.mode_set(mode='OBJECT')

            obj.data.vertices[m].select = True
            obj.data.vertices[n].select = True

            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.subdivide()

            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.mode_set(mode='OBJECT')

            obj.data.vertices[last_dot].select = True
            obj.data.vertices[n].select = True

            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.merge(type='CENTER')

        bpy.ops.object.mode_set(mode='OBJECT')


    def export_gltf(self):
        r1 = self.file_name.split('/')
        save_dir = ""
        for idx in range(len(r1) - 2):
            save_dir = save_dir + r1[idx] + "/"

        save_dir += '3D/'

        s_file_name = r1[-1]
        s_file_name = s_file_name.split('.')[0]

        bpy.ops.export_scene.gltf(filepath=save_dir+s_file_name+".glb", export_format="GLB")
        # bpy.ops.wm.save_as_mainfile(filepath=save_dir+s_file_name+".blend")
        print("export : ", save_dir+s_file_name+".gltf" )

    #도형 생성(점)
    def create_visage_plane(self,list, object_name):
        mesh_size = 468
        bpy.ops.mesh.primitive_circle_add(vertices=mesh_size, radius=0.1, enter_editmode=False, location=(0, 0, 0))
        obj = bpy.data.objects["Circle"]
        obj.name = object_name
        vertex_list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]

        obj = bpy.data.objects[object_name]
        # select vertex
        obj = bpy.context.view_layer.objects.active

        for i in range(mesh_size):
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.mode_set(mode='OBJECT')

            obj.data.vertices[i].select = True

            bpy.ops.object.mode_set(mode='EDIT')
            diff_x = list[i][0] - vertex_list[i][0]
            diff_y = list[i][1] - vertex_list[i][1]
            diff_z = list[i][2] - vertex_list[i][2]

            bpy.ops.transform.translate(value=(diff_x, diff_y, diff_z))
        bpy.ops.object.mode_set(mode='OBJECT')
    def create_piece_plane(self,list, piece_index, object_name):
        mesh_size = len(piece_index)
        bpy.ops.mesh.primitive_circle_add(vertices=mesh_size, radius=0.1, enter_editmode=False, location=(0, 0, 0))
        obj = bpy.data.objects["Circle"]
        obj.name = object_name
        vertex_list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]

        obj = bpy.data.objects[object_name]
        # select vertex
        obj = bpy.context.view_layer.objects.active

        for i in range(mesh_size):
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.mode_set(mode='OBJECT')

            obj.data.vertices[i].select = True

            bpy.ops.object.mode_set(mode='EDIT')
            diff_x = list[piece_index[i]][0] - vertex_list[i][0]
            diff_y = list[piece_index[i]][1] - vertex_list[i][1]
            diff_z = list[piece_index[i]][2] - vertex_list[i][2]

            bpy.ops.transform.translate(value=(diff_x, diff_y, diff_z))
        bpy.ops.object.mode_set(mode='OBJECT')
    #예외가 있는 createFace
    def create_visage_Face(self,list, object_name, excption_list):
        self.chooes_object(object_name)
        obj = bpy.data.objects[object_name]
        # select vertex
        obj = bpy.context.view_layer.objects.active

        for i in list:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.mode_set(mode='OBJECT')

            if i[0] not in excption_list or i[1] not in excption_list or i[2] not in excption_list:
                obj.data.vertices[i[0]].select = True
                obj.data.vertices[i[1]].select = True
                obj.data.vertices[i[2]].select = True

            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.edge_face_add()

        bpy.ops.object.mode_set(mode='OBJECT')

    def create_piece_Face(self,list, piece_list, object_name, exception_list):
        self.chooes_object(object_name)
        obj = bpy.data.objects[object_name]
        # select vertex
        obj = bpy.context.view_layer.objects.active

        for i in list:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.mode_set(mode='OBJECT')

            if i[0] in piece_list and i[1] in piece_list and i[2] in piece_list:
                if i[0] not in exception_list or i[1] not in exception_list or i[2] not in exception_list:
                    obj.data.vertices[piece_list.index(i[0])].select = True
                    obj.data.vertices[piece_list.index(i[1])].select = True
                    obj.data.vertices[piece_list.index(i[2])].select = True

            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.edge_face_add()

        bpy.ops.object.mode_set(mode='OBJECT')
    def create_Iris(self,list, piece_list,object_name):
        mesh_size = (len(piece_list)-1) * 2
        center = list[piece_list[0]]
        sum = 0
        for i in range(1,len(piece_list)):
            sum = sum + abs(list[piece_list[i]][0] - center[0])
            sum = sum + abs(list[piece_list[i]][1] - center[1])
            sum = sum + abs(list[piece_list[i]][2] - center[2])
        radius = sum / (len(piece_list)-1)

        bpy.ops.mesh.primitive_circle_add(vertices=mesh_size, radius=radius, enter_editmode=False, location=center)
        obj = bpy.data.objects["Circle"]
        obj.name = object_name
        vertex_list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')

        for i in range(mesh_size):
            obj.data.vertices[i].select = True

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.edge_face_add()
        bpy.ops.object.mode_set(mode='OBJECT')

    def remove_visage_Edge(self,expt_list, object_name):
        self.chooes_object(object_name)
        obj = bpy.data.objects[object_name]
        # select vertex
        obj = bpy.context.view_layer.objects.active
        vertex_list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='EDGE')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')

        for i in range(len(vertex_list)):
            if i not in expt_list:
                obj.data.edges[i].select = True

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.delete(type='EDGE')

        bpy.ops.object.mode_set(mode='OBJECT')
    def remove_piece_Edge(self,delete_list, object_name):
        self.chooes_object(object_name)
        obj = bpy.data.objects[object_name]
        # select vertex
        obj = bpy.context.view_layer.objects.active
        vertex_list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='EDGE')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')

        for i in range(len(vertex_list)):
            if i not in delete_list:
                obj.data.edges[i].select = True

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.delete(type='EDGE')

        bpy.ops.object.mode_set(mode='OBJECT')
    def shade_smooth(self,object_name):
        self.chooes_object(object_name)
        obj = bpy.data.objects[object_name]
        # select vertex
        obj = bpy.context.view_layer.objects.active

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='VERT')

        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.faces_shade_smooth()
        bpy.ops.mesh.normals_make_consistent(inside=False)

        bpy.ops.object.mode_set(mode='OBJECT')
    #랜드마크 리스트 생성
    def create_landmarks_list(self):
        list =[]
        for facial_landmarks in self.result.multi_face_landmarks:

            for i in range(0, 478):
                pt1 = facial_landmarks.landmark[i]
                list.append((pt1.x-0.5, pt1.y-0.5, pt1.z-0.5))

        return list
    #필요 없는 선 list
    def create_remove_exception_Edge_list(self,Face_list, exception_list):
        list =[]

        for i in Face_list:
            if i[0] not in exception_list and i[1] not in exception_list and i[2] not in exception_list:
                if abs(i[0] - i[1]) == 1 :
                    index = min([i[0], i[1]])
                    list.append(index)
                elif abs(i[1] - i[2]) == 1:
                    index = min([i[1], i[2]])
                    list.append(index)
                elif abs(i[2] - i[0]) == 1:
                    index = min([i[0], i[2]])
                    list.append(index)
        return list
    def create_remove_Edge_list(self,Face_list, piece_list, exception_list):
        list =[]

        for i in Face_list:
            if i[0] in piece_list and i[1] in piece_list and i[2] in piece_list:
                if i[0] not in exception_list or i[1] not in exception_list or i[2] not in exception_list:
                    a0 = piece_list.index(i[0])
                    a1 = piece_list.index(i[1])
                    a2 = piece_list.index(i[2])
                    if abs(a0 - a1) == 1 :
                        index = min(a0, a1)
                        list.append(index)
                    elif abs(a1 - a2) == 1 :
                        index = min(a1, a2)
                        list.append(index)
                    elif abs(a2 - a0) == 1 :
                        index = min(a2, a0)
                        list.append(index)

        return list
    #랜드마크리스트 좌우대칭
    def relocation_list(self):
        middle_index = []
        right_index = []
        left_index = []

        for i in range(int(len(OUTLINE_POINTS_1) / 2)):
            if i == 0:
                middle_index.append(OUTLINE_POINTS_1[i])
                middle_index.append(OUTLINE_POINTS_1[i + int(len(OUTLINE_POINTS_1) / 2)])
            else:
                right_index.append(OUTLINE_POINTS_1[i])
                left_index.append(OUTLINE_POINTS_1[len(OUTLINE_POINTS_1) - i])

        for i in range(int(len(OUTLINE_POINTS_2) / 2)):
            if i == 0:
                middle_index.append(OUTLINE_POINTS_2[i])
                middle_index.append(OUTLINE_POINTS_2[i + int(len(OUTLINE_POINTS_2) / 2)])
            else:
                right_index.append(OUTLINE_POINTS_2[i])
                left_index.append(OUTLINE_POINTS_2[len(OUTLINE_POINTS_2) - i])

        for i in range(int(len(OUTLINE_POINTS_3) / 2)):
            if i == 0:
                middle_index.append(OUTLINE_POINTS_3[i])
                middle_index.append(OUTLINE_POINTS_3[i + int(len(OUTLINE_POINTS_3) / 2)])
            else:
                right_index.append(OUTLINE_POINTS_3[i])
                left_index.append(OUTLINE_POINTS_3[len(OUTLINE_POINTS_3) - i])

        for i in range(int(len(OUTLINE_POINTS_4) / 2)):
            if i == 0:
                middle_index.append(OUTLINE_POINTS_4[i])
                middle_index.append(OUTLINE_POINTS_4[i + int(len(OUTLINE_POINTS_4) / 2)])
            else:
                right_index.append(OUTLINE_POINTS_4[i])
                left_index.append(OUTLINE_POINTS_4[len(OUTLINE_POINTS_4) - i])

        for i in RIGHT_OUTLINE_POINT:
            right_index.append(i)

        for i in RIGHT_EYE_POINT_1:
            right_index.append(i)

        for i in RIGHT_EYE_POINT_2:
            right_index.append(i)

        for i in RIGHT_EYE_POINT_3:
            right_index.append(i)

        for i in LEFT_OUTLINE_POINT:
            left_index.append(i)

        for i in range(len(LEFT_EYE_POINT_1)):
            if i <= 8:
                left_index.append(LEFT_EYE_POINT_1[8 - i])
            else:
                left_index.append(LEFT_EYE_POINT_1[24 - i])

        for i in range(len(LEFT_EYE_POINT_2)):
            if i <= 8:
                left_index.append(LEFT_EYE_POINT_2[8 - i])
            else:
                left_index.append(LEFT_EYE_POINT_2[24 - i])

        for i in range(len(LEFT_EYE_POINT_3)):
            if i <= 8:
                left_index.append(LEFT_EYE_POINT_3[8 - i])
            else:
                left_index.append(LEFT_EYE_POINT_3[24 - i])

        # 입
        for i in range(int(len(MOUSE_POINT_1) / 2)):
            if i == 5:
                middle_index.append(MOUSE_POINT_1[i])
                middle_index.append(MOUSE_POINT_1[i + int(len(MOUSE_POINT_1) / 2)])
            elif i < 5:
                right_index.append(MOUSE_POINT_1[i])
                left_index.append(MOUSE_POINT_1[10 - i])
            else:
                right_index.append(MOUSE_POINT_1[25 - i])
                left_index.append(MOUSE_POINT_1[5 + i])

        for i in range(int(len(MOUSE_POINT_2) / 2)):
            if i == 5:
                middle_index.append(MOUSE_POINT_2[i])
                middle_index.append(MOUSE_POINT_2[i + int(len(MOUSE_POINT_2) / 2)])
            elif i < 5:
                right_index.append(MOUSE_POINT_2[i])
                left_index.append(MOUSE_POINT_2[10 - i])
            else:
                right_index.append(MOUSE_POINT_2[25 - i])
                left_index.append(MOUSE_POINT_2[5 + i])

        for i in range(int(len(MOUSE_POINT_3) / 2)):
            if i == 5:
                middle_index.append(MOUSE_POINT_3[i])
                middle_index.append(MOUSE_POINT_3[i + int(len(MOUSE_POINT_3) / 2)])
            elif i < 5:
                right_index.append(MOUSE_POINT_3[i])
                left_index.append(MOUSE_POINT_3[10 - i])
            else:
                right_index.append(MOUSE_POINT_3[25 - i])
                left_index.append(MOUSE_POINT_3[5 + i])

        for i in range(int(len(MOUSE_POINT_4) / 2)):
            if i == 5:
                middle_index.append(MOUSE_POINT_4[i])
                middle_index.append(MOUSE_POINT_4[i + int(len(MOUSE_POINT_4) / 2)])
            elif i < 5:
                right_index.append(MOUSE_POINT_4[i])
                left_index.append(MOUSE_POINT_4[10 - i])
            else:
                right_index.append(MOUSE_POINT_4[25 - i])
                left_index.append(MOUSE_POINT_4[5 + i])

        for i in range(int(len(MOUSE_POINT_5) / 2)):
            if i == 5:
                middle_index.append(MOUSE_POINT_5[i])
                middle_index.append(MOUSE_POINT_5[i + int(len(MOUSE_POINT_5) / 2)])
            elif i < 5:
                right_index.append(MOUSE_POINT_5[i])
                left_index.append(MOUSE_POINT_5[10 - i])
            else:
                right_index.append(MOUSE_POINT_5[25 - i])
                left_index.append(MOUSE_POINT_5[5 + i])

        for i in range(int(len(NOSE_POINT_1) / 2)):
            if i == 0:
                middle_index.append(NOSE_POINT_1[i])
                middle_index.append(NOSE_POINT_1[i + int(len(NOSE_POINT_1) / 2)])
            else:
                right_index.append(NOSE_POINT_1[i])
                left_index.append(NOSE_POINT_1[len(NOSE_POINT_1) - i])

        for i in MIDDLE_NOSE_LINE:
            middle_index.append(i)

        for i in RIGHT_NOSE_POINT:
            right_index.append(i)

        for i in LEFT_NOSE_POINT:
            left_index.append(i)

        for i in RIGHT_IRIS:
            right_index.append(i)

        for i in range(len(LEFT_IRIS)):
            if i % 2 == 1:
                left_index.append(LEFT_IRIS[4-i])
            else:
                left_index.append(LEFT_IRIS[i])



        return middle_index, right_index, left_index
    #실제 좌표 좌우대칭
    def symmetry_list(self,landmarks_list,middle_index, right_index, left_index):
        list = landmarks_list

        mid_x = 0

        for i in middle_index:
            list[i] = (mid_x, list[i][1], list[i][2])

        for i in range(len(left_index)):
            mid_y = (list[right_index[i]][1] + list[left_index[i]][1])/2
            mid_z = (list[right_index[i]][2] + list[left_index[i]][2])/2

            distance_x = (list[left_index[i]][0] - list[right_index[i]][0])/2
            list[right_index[i]] = (mid_x-distance_x, mid_y, mid_z)
            list[left_index[i]] = (mid_x+distance_x, mid_y, mid_z)

        return list
    def objects_join(self,objects_list):
        print(objects_list)
        bpy.ops.object.select_all(action='DESELECT')
        self.chooes_object(objects_list[0])
        for i in objects_list:
            bpy.ops.object.select_pattern(pattern=i)
        bpy.ops.object.join()

    def move_origin_center(self,object_name):
        self.chooes_object(object_name)
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
        bpy.context.object.location = (0, 0, 0)

    def move_origin_coord(self, object_name, coord):
        self.chooes_object(object_name)
        bpy.context.scene.cursor.location = (coord[0], coord[1], coord[2])
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
        bpy.context.object.location = (0, 0, 0)

    def rotation_object(self,object_name):
        self.chooes_object(object_name)
        bpy.context.object.rotation_euler[0] = 1.5708
        bpy.context.object.rotation_euler[1] = 1.5708 * 2
        bpy.context.object.rotation_euler[2] = 1.5708 * 2

    def rotation_axis(self, object_name):

        self.chooes_object(object_name)
        obj = bpy.data.objects['Visage']
        list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]

    def skin_material(self, object_name):
        self.chooes_object(object_name)
        bpy.ops.object.material_slot_remove()
        new_mat = bpy.data.materials.new("skin")
        new_mat.diffuse_color = (1, 0.730394, 0.502537, 1)
        bpy.ops.object.material_slot_add()
        bpy.context.object.active_material = new_mat

    def get_linkPoint(self, object_name = "Visage"):
        self.chooes_object(object_name)
        obj = bpy.context.view_layer.objects.active
        vertex_list = [(obj.matrix_world @ v.co) for v in obj.data.vertices]

        print(len(vertex_list))
        print(vertex_list[742], vertex_list[757], vertex_list[729])

        return vertex_list[742], vertex_list[757], vertex_list[729]

    def resize_scale_y(self, object_name, value = 1.05):
        self.chooes_object(object_name)
        obj = bpy.context.view_layer.objects.active
        obj.scale[1] *= value
        if self.expansion_ratio != None:
            self.expansion_ratio = (self.expansion_ratio[0], self.expansion_ratio[1]*value, self.expansion_ratio[2])


    def createMesh(self):
        #리스트들 선언
        visage_create_expt_list = MOUSE_POINT_1 + MOUSE_POINT_2 + MOUSE_POINT_3 + MOUSE_POINT_4 + LEFT_EYE_POINT_1 + RIGHT_EYE_POINT_1 + NOSTRILL_POINT
        visage_remove_expt_list_2 = MOUSE_POINT_1 + MOUSE_POINT_2 + MOUSE_POINT_3
        mouse_list = sorted(MOUSE_POINT_1 + MOUSE_POINT_2 + MOUSE_POINT_3 + MOUSE_POINT_4)

        self.delete_all_object()

        landmarks_point = self.create_landmarks_list()
        middle_index, right_index, left_index = self.relocation_list()
        landmarks_point = self.symmetry_list(landmarks_point, middle_index, right_index, left_index)

        visage_edge_expt_list = self.create_remove_exception_Edge_list(Face_index, visage_remove_expt_list_2)
        mouse_edge_remove_list = self.create_remove_Edge_list(Face_index, mouse_list, MOUSE_POINT_1)


        self.create_visage_plane(landmarks_point,"Visage")
        self.create_visage_Face(Face_index,"Visage", visage_create_expt_list)
        self.remove_visage_Edge(visage_edge_expt_list,"Visage")
        # self.shade_smooth("Visage")

        self.create_piece_plane(landmarks_point, mouse_list,"Mouse")
        self.create_piece_Face(Face_index, mouse_list, "Mouse", MOUSE_POINT_1)
        self.remove_piece_Edge(mouse_edge_remove_list, "Mouse")
        # self.shade_smooth("Mouse")

        self.create_piece_plane(landmarks_point, RIGHT_EYE_POINT_1,"Right_Eyes")
        self.create_piece_Face(Face_index, RIGHT_EYE_POINT_1, "Right_Eyes", [])
        # self.shade_smooth("Right_Eyes")

        self.create_piece_plane(landmarks_point, LEFT_EYE_POINT_1,"Left_Eyes")
        self.create_piece_Face(Face_index, LEFT_EYE_POINT_1, "Left_Eyes", [])
        # self.shade_smooth("Left_Eyes")



        self.create_Iris(landmarks_point,RIGHT_IRIS,"Right_iris")


        self.create_Iris(landmarks_point,LEFT_IRIS,"Left_iris")

        self.objects_join(["Visage","Mouse","Right_Eyes","Left_Eyes","Right_iris","Left_iris"])

        self.rotation_object("Visage")

        self.import_head()
        self.move_origin_center("headpart")

        self.rotation_axis("Visage")

        self.resize_head(landmarks_point)
        self.combine_head()
        self.resize_scale_y("Visage", value= 1.0)
        self.move_origin_center("Visage")

        front_point, right_point, back_point = self.get_linkPoint()

        self.export_gltf()


        return [self.expansion_ratio , front_point, right_point, back_point]